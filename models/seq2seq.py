"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Sequence-To-Sequence (Seq2Seq) model definition.

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

import random
from typing import Any, Dict, List
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.utils import gpu


class Seq2SeqNet(nn.Module):

    def __init__(self, config):
        super(Seq2SeqNet, self).__init__()
        self.config = config
        self.embedding_size = config.model.embedding_size
        self.hidden_size = config.model.hidden_size
        self.output_latent = config.model.output_latent

        self.enc = EncoderRNN(embedding_size=self.embedding_size, hidden_size=self.hidden_size)
        self.dec = DecoderRNN(embedding_size=self.embedding_size, hidden_size=self.hidden_size)
        self.lstm_enc_dec = Seq2Seq(self.enc, self.dec, self.config.dataset.seq_len)
        
        self.output = dict()


    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:

        rot = gpu(data["rot"])
        orig = gpu(data["orig"])
        gt_preds = gpu(data["gt_preds"])

        future_target_global = gt_preds
        future_target_global = torch.stack([x[0] for x in future_target_global])
        future_target_local = gpu(torch.zeros_like(future_target_global))

        # For the linear residual decoder in lanegcn it does not matter
        for i in range(len(future_target_global)):
            future_target_local[i] = torch.matmul(rot[i], (future_target_global[i] - orig[i]).transpose(0,1)).transpose(0,1)
        
        # We must flip the output sequence for the learning part and then flip it back, before writing into the "reg" list. Because origin is last timestep
        future_target_local_flipped = future_target_local.flip([1])

        if self.config.model.use_deltas:
            feats = gpu(data["feats"])
            feats_target = torch.stack([x[0][:,:2] for x in feats])
        else:
            feats_target = future_target_local
        

        if self.output_latent:
            out, latent = self.lstm_enc_dec(feats_target, future_target_local_flipped, output_latent=True)
        else:
            out = self.lstm_enc_dec(feats_target, future_target_local_flipped, output_latent=False)

        out_flipped = out.flip([1])
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
        rot, orig = gpu(data["rot"]), gpu(data["orig"])

        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )

        if self.output_latent:
            out["latent_features"] = [vec[None, :] for vec in latent] 

        out["gt_preds"] = data["gt_preds"]

        self.output = out

        return out

class EncoderRNN(nn.Module):
    """
    Encoder Network.
    
    """

    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 16):
        """
        Initialize the encoder network.
        Args:
            input_size: number of features in the input
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
        """

        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:
        """
        Run forward propagation.
        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 

        """

        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
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
    def __init__(self, encoder, decoder, rollout_length):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.rollout_length = rollout_length
        
  
    def forward(self, src, trg, teacher_forcing_ratio = 0.0, output_latent=False):
        
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = src.shape[0]
        input_length = src.shape[1]

        output_length = trg.shape[1]

        encoder_hidden = (
            gpu(torch.zeros(batch_size, self.encoder.hidden_size)),
            gpu(torch.zeros(batch_size, self.encoder.hidden_size))
        )

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = src[:, ei, :]
            encoder_hidden = self.encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = gpu(torch.zeros(trg.shape))

        # Decode hidden state in future trajectory
        for di in range(self.rollout_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output
            
            #decide if we are going to use teacher forcing or not
            if self.training:
                teacher_force = random.random() < teacher_forcing_ratio
            else:
                teacher_force = False

            # Use own predictions as inputs at next step
            decoder_input = trg[:, di,:] if teacher_force else decoder_output
        
        if output_latent:
            latent_features = encoder_hidden[0]
            return decoder_outputs, latent_features
        else:
            return decoder_outputs
