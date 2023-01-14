"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Spatio-Temporal Graph Auto-Encoder (STGAE) model definition.

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

import numpy as np
import random
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops
from typing import Any, Dict, List

from src.utils import gpu


class STGAE(nn.Module):
    """
    Spatio-temporal Seq2Seq LSTM.

    """

    def __init__(self, config):
        super(STGAE, self).__init__()
        self.config = config
        self.embedding_size = config.model.embedding_size
        self.hidden_size = config.model.hidden_size
        self.output_latent = config.model.output_latent

        self.encs = EncoderSpatial(embedding_size=self.embedding_size)
        self.enct = EncoderRNN(input_size=self.embedding_size, hidden_size=self.hidden_size)
        self.dec = DecoderRNN(embedding_size=self.embedding_size, hidden_size=self.hidden_size)
        self.lstm_enc_dec = Seq2Seq(self.encs, self.enct, self.dec, self.config.dataset.seq_len)
    
        self.output = dict()


    def forward(self, data: Dict, output_latent=True) -> Dict[str, List[Tensor]]:

        rot = gpu(data["rot"])
        orig = gpu(data["orig"])
        gt_preds = gpu(data["gt_preds"])

        future_target_global = gt_preds
        future_target_local = [gpu(torch.zeros_like(ftg)) for ftg in future_target_global]

        # Transform the trajectories of all actors, i.e.
        # translation: substract the origin of reference actor and 
        # rotation: rotate by the rotation of the reference actor (dirving direction in first two frames)
        for i in range(len(future_target_global)):
            ftg = future_target_global[i]
            N = ftg.shape[0]
            rot_mat = rot[i].repeat(1, N, 1, 1)
            trans_trajs = ftg - orig[i]
            future_target_local[i] = torch.matmul(rot_mat, trans_trajs.transpose(1, 2)).transpose(2, 3)[0]

        future_target_local_flipped = [torch.flip(ftl[0], [0]) for ftl in future_target_local]
        future_target_local_flipped = torch.stack(future_target_local_flipped)

        if self.config.model.use_deltas:
            feats = gpu(data["feats"])
            feats_target = torch.stack([x[0][:,:2] for x in feats])
        else:
            feats_target = future_target_local
        
        if output_latent:
            dec_out, latent = self.lstm_enc_dec(feats_target, future_target_local_flipped, output_latent=True)
        else:
            dec_out = self.lstm_enc_dec(feats_target, future_target_local_flipped)

        # Node Regression Output
        reg_out = dec_out["reg"]
    
        out_flipped = reg_out.flip([1])
        out_flipped = out_flipped.unsqueeze(1).unsqueeze(1)

        out = dict()
        out["reg"] = []

        # create list of tensors and pad dummy predictions for all remaining agents
        for b in range(out_flipped.shape[0]):
            N = gt_preds[b].shape[0]
            reg_shape = [N] + list(out_flipped.shape[2:])
            reg = gpu(torch.zeros(reg_shape))
            reg[0, 0] = out_flipped[b]
            out["reg"].append(reg)

        # prediction
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )

        if self.output_latent:
            out["latent_features"] = [vec[None, :] for vec in latent]

        out["gt_preds"] = data["gt_preds"]
        self.output = out

        return out


class EncoderSpatial(nn.Module):
    """
    Spatial encoder network.

    """

    def __init__(self, 
                 input_size: int = 2,
                 embedding_size: int = 8,
                 ):
        """Initialize spatial encoder network.
        Args:
            input_size: number of features in the input.
            embedding_size: number of embedding units.
            encoding: type of encoding mechanism, e.g. linear or graphical etc.    
        """
        super(EncoderSpatial, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size

        # parameter for distance weights
        self.distance_norm = 1  #m
        self.coord_indices = {"dx": 0, "dxnonorm": 0, "dy": 1, "dynonorm": 1}

        # Residuals
        self.residual = nn.Sequential(
                nn.Conv2d(
                    self.input_size,
                    self.embedding_size,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(self.embedding_size),
            )

        self.residual = nn.Linear(self.input_size, self.embedding_size, bias=False)
        self.embedding_layer = GCNConv(in_channels=self.input_size, out_channels=self.embedding_size, add_self_loops=True)

    
    def forward(self, x: torch.FloatTensor):
        """Forward path throuth the spatial encoder network.
        Args:
            x: List of input features. [batch_size] x [N_actors x T x D]
        """

        batch_size = len(x)
        seq_len = x[0].shape[1]
        feat_dim = x[0].shape[2]
        inputs = gpu(torch.zeros(batch_size, seq_len, feat_dim))
        enc_sp = gpu(torch.zeros((batch_size, seq_len, self.embedding_size)))
        agents_per_sample = [xi.shape[0] for xi in x]

        # data preprocessing (reshaping)
        nT = x[0].shape[1]
        nF = x[0].shape[2]

        graph_input, edge_index, edge_weight = self.preprocess_data_spatio_temporal_graph(x, agents_per_sample, T=nT, F=nF)

        # linear residuals
        res = self.residual(graph_input)

        # one-dimensional edge weights [|E|]
        edge_weight = edge_weight[:, 0]

        # encoding
        enc = self.embedding_layer(graph_input, edge_index=edge_index, edge_weight=edge_weight)

        # Add residuals
        enc = enc + res

        # non-linearity
        enc = F.relu(enc)

        # output post-processing (reshaping)
        nF = enc.shape[1]
        enc_sp = self.postprocess_data_spatio_temporal_graph(enc, agents_per_sample, T=nT, F=nF)

        # inputs
        inputs = [xi[0][None, :] for xi in x]
        inputs = torch.cat(inputs)

        # output embedding of agent of interest
        return inputs, enc_sp


    def preprocess_data_spatio_temporal_graph(self, data, agents_per_sample, T=16, F=2):
        """ Preprocess data for the computation in a spatio-temporal graph

        Args:
            data: list of samples of shape[B] x [N x T x F] with variable N
            agents_per_sample: list of number of agents per sample [B]
            T: number of time steps per sample, default T=16
            F: number of features per sample, e.g. x- and y-location, default F=2
        
        Out:
            x: input data for the spatio-temporal graph [M x F] with M being the number of all agents times the nubmer of time steps
            edge_indices: the connection indices of the graph [2 x |E|] with E being the number of edges in all graphs
            edge_weights (optional): precomputed edge weights [|E|]
        """

        # data
        data_swap = [torch.swapaxes(sample, 0, 1) for sample in data]

        data_reshape = [torch.reshape(sample, (-1, F)) for sample in data_swap]

        x = torch.cat(data_reshape)

        # edge indices
        edge_index = []
        offset = 0

        for N in agents_per_sample:

            for t in range(T):

                # fully-connected
                ei = np.arange(N).repeat(N)
                ej = np.tile(np.arange(N), N)
                edge_indices = np.stack((ei, ej)) + offset
                edge_indices = remove_self_loops(edge_indices)[0]

                # offset the indices list
                offset += N

                # edge index
                edge_index.append(edge_indices)

        edge_index = np.column_stack(edge_index)

        # edge weights
        edge_weight = []

        edge_weight.append(self.distance_weights(x, edge_index, type="distance"))
            
        edge_weight = torch.cat(edge_weight, axis=1)
        edge_index = gpu(torch.tensor(edge_index))

        return x, edge_index, edge_weight


    def distance_weights(self, x, edge_index, type="distance"):
        """Computes the distance weights based on the data matrix x [N x F] and the edge index matrix [2, |E|]
        
        Attr:
            x:               data matrix [N x F]
            edge_index:      edge index matrix [2, |E|]
            type:            type of distance edge, e.g. distance, distance_squared, distance_nonorm

        Return:
            edge_weight:    edge weights as cuda torch tensor [|E|, 1]

        """
        
        #TODO: Keep in mind that this only works, as long as "use_deltas" == False
        # distance based edge weights
        edge_index = gpu(torch.tensor(edge_index))
        edge_index = remove_self_loops(edge_index)[0]
        vi = x[edge_index[0]]
        vj = x[edge_index[1]]
        delta = vi - vj
        distance = torch.linalg.norm(delta, axis=1)
        distance = torch.where(distance == 0.0, gpu(torch.tensor(1e-6)), distance)
        if type == "distance_squared":
            distance = torch.square(distance)

        # normalize
        distance = distance[:, None]
        if type != "distance_nonorm":
            distance = torch.divide(torch.tensor(self.distance_norm), distance)
            distance[distance > 1.0] = 1.0
        edge_weight = gpu(distance)

        return edge_weight


    def postprocess_data_spatio_temporal_graph(self, data, agents_per_sample, T=16, F=8, select_agent_only=True):

        data_points_per_sample = [agents_per_sample[i] * T for i in range(len(agents_per_sample))]

        out_reshape = torch.split(data, data_points_per_sample)
        out = [torch.reshape(outi, (T, agents_per_sample[i], F)) for i, outi in enumerate(out_reshape)]
        out = [torch.swapaxes(outi, 0, 1) for outi in out]

        if select_agent_only:
            out = torch.cat([outi[0][None, :] for outi in out])
        else:
            out = torch.cat(out)

        return out


class EncoderRNN(nn.Module):
    """
    Encoder Network.

    """

    def __init__(self,
                 input_size: int = 8,
                 hidden_size: int = 16):
        """Initialize the encoder network.
        Args:
            input_size: number of features in the input
            hidden_size: Hidden size of LSTM
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:
        """Run forward propagation.
        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 
        """
        hidden = self.lstm1(x, hidden)
        return hidden


class DecoderRNN(nn.Module):
    """
    Decoder Network.
    
    """

    def __init__(self, embedding_size=8, hidden_size=16, output_size=2):
        """
        Initialize the decoder network.
        Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output
        """

        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        Run forward propagation.
        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            output: output from lstm
            hidden: final hidden state
        """
        
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, embedding, encoder, decoder, rollout_length, structure_decoder=None):
        super().__init__()
        
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.structure_decoder = structure_decoder
        self.rollout_length = rollout_length
        
  
    def forward(self, src, trg, teacher_forcing_ratio = 0.0, output_latent=False):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = len(src)
        input_length = src[0].shape[1]
        output_length = trg[0].shape[0]

        # spatial encoding
        inputs, embeddings = self.embedding(src)

        # temporal encoding
        encoder_hidden = (
            gpu(torch.zeros(batch_size, self.encoder.hidden_size)),
            gpu(torch.zeros(batch_size, self.encoder.hidden_size))
        )

        for ei in range(input_length):
            encoder_input = embeddings[:, ei, :]
            encoder_hidden = self.encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = inputs[:,-1, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_node_output = gpu(torch.zeros(trg.shape))

        # Decode hidden state in future trajectory
        for di in range(self.rollout_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                     decoder_hidden)
            decoder_node_output[:, di, :] = decoder_output
            
            #decide if we are going to use teacher forcing or not
            if self.training:
                teacher_force = random.random() < teacher_forcing_ratio
            else:
                teacher_force = False

            # Use own predictions as inputs at next step
            decoder_input = trg[:, di,:] if teacher_force else decoder_output
        
        # log output
        decoder_output = dict()
        decoder_output["reg"] = decoder_node_output

        # Structural decoding
        if self.structure_decoder:
            decoder_struct_input = encoder_hidden[0]
            decoder_struct_output = self.structure_decoder(decoder_struct_input)

            decoder_output["struct_reg"] = decoder_struct_output

        # output latent features
        if output_latent:
            latent_features = encoder_hidden[0]
            return decoder_output, latent_features
        else:
            return decoder_output
