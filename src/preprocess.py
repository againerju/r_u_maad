"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Data pre-processing. Parts from the original LaneGCN are indicated in the comments. 

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

import os
import sys
import pandas as pd
import pickle
import parmap
import random
import more_itertools
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import gpu
from src.dataset import ArgoDataset, collate_fn, from_numpy

random.seed(42)


def sliding_window_preprocess(input_path, output_path, n_cpus, n_samples=100, size=16, stride=-1):
    """ 
    Given the *.csv files in the input_path, apply a sliding window algorithm to create sequences of equal length.

    Arguments:
        input_path      path to raw .csv files
        output_path     path to processed sliding window .csv files
        n_cpus          numer of cpus used for parallelization
        n_samples       number of samples selected from the raw .csv files, -1 uses all files
        size            sliding window width, i.e. sequence length
        stride          sliding window stride, i.e. 1 creates one sub-sequence per time step, 
                        -1 randomly selects one sub-sequence from the original sequence

    """

    # Read the files and sort them
    # (the output of os.listdir differs on different systems)
    input_files = sorted(os.listdir(input_path))

    # Create absolut paths
    input_files = [os.path.join(input_path, x) for x in input_files]

    # If this parameter is -1, we take all the files in the folder
    if n_samples == -1:
        n_samples = len(input_files)

    # Create the output folder, if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    assert len(input_files) > 0, "No files found"

    # Randomly shuffle the files (with defined seed)
    random.shuffle(input_files)

    # Create parameters for each file
    file_params = []
    print("Creating file information")
    for current_input_path in input_files[:n_samples]:

        # Generate full output path
        current_output_path = os.path.join(
            output_path, os.path.basename(current_input_path)
        )

        current_file_params = {
            "input_path": current_input_path,
            "output_path": current_output_path,
            "length": size,
            "stride": stride,
        }

        file_params.append(current_file_params)

    print("Starting to preprocess")
    parallel_out = parmap.map(sliding_window, file_params,
                                pm_processes=n_cpus, pm_pbar=True)


def sliding_window(kwargs):
    """Given the kwargs, create a sequence using the sliding window method.
    
    """

    # Arguments
    input_path = kwargs["input_path"]
    output_path = kwargs["output_path"]
    length = kwargs["length"]
    stride = kwargs["stride"]

    # Read dataframe
    seq_df = pd.read_csv(input_path)

    # Select time stamps where AV and AGENT/CONTROLLED are present
    object_types = seq_df["OBJECT_TYPE"].unique()

    assert "AV" in object_types, "AV does not exist"
    assert any(k in object_types for k in ["AGENT", "CONTROLLED"]), "Neither 'AGENT' nor 'CONTROLLED' exists"

    timestamps = []
        
    for type in ["AV", "AGENT", "CONTROLLED"]:
        
        if type in object_types:
            timestamps.append(seq_df[seq_df["OBJECT_TYPE"] == type]["TIMESTAMP"])
    
    valid_timestamps = timestamps[0]

    for i in range(1, len(timestamps)):

        valid_timestamps = set(valid_timestamps) & set(timestamps[i])
    
    seq_df = seq_df[seq_df["TIMESTAMP"].isin(valid_timestamps)]

    # Get all unique TIMESTAMPs
    timestamps = sorted(seq_df["TIMESTAMP"].unique())

    # Sort the dataframe given TIMESTAMP and OBJECT_TYPE
    seq_df = seq_df.sort_values(by=["TIMESTAMP", "OBJECT_TYPE"], key=sorter).reset_index(drop=True)

    # If this parameter is -1, we take on random window with length
    if stride == -1:
        random_start = random.randint(0, len(timestamps) - length)
        out_df = seq_df[seq_df["TIMESTAMP"].isin(timestamps[random_start : random_start+length])]
        out_df.to_csv(os.path.splitext(output_path)[0] + ".csv")
    else:
        windows = list(more_itertools.windowed(timestamps, n=length, step=stride))

        for i, window in enumerate(windows):
            if None in window:
                break

            out_df = seq_df[seq_df["TIMESTAMP"].isin(window)]

            last_ts = int(max(out_df.FRAME_ID.unique()))

            out_df.to_csv(os.path.splitext(output_path)[0] + "_ts_{:03}".format(last_ts) + ".csv")


# Inspired by https://stackoverflow.com/questions/23482668/sorting-by-a-custom-list-in-pandas
def sorter(column):
    if column.name == "OBJECT_TYPE":
        """Sort function"""
        types = ["AV", "CONTROLLED", "AGENT", "OTHERS"]
        correspondence = {type: order for order, type in enumerate(types)}
        return column.map(correspondence)
    else:
        return column


def pickle_preprocess(config, input_path, output_path, phase="train"):
    """
    Combine the data of all .csv files into one pickle file inlcuding graph structured data. 

    """

    # config
    if phase in ["train", "test"]:
        train = True
    elif phase == "val":
        train = False
    else:
        sys.exit(f"Unkonwn set {config.preprocess.set}. Abort!")

    # dataloader
    dataset = ArgoDataset(input_path, config, train=train, seq_len=config.dataset.seq_len)
    train_loader = DataLoader(
        dataset,
        batch_size=config.preprocess.batch_size,
        num_workers=config.preprocess.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # create pre-processing dataset given the data keys
    stores = [None for x in range(len(dataset.avl.seq_list))]
    for i, data in enumerate(tqdm(train_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in get_data_keys(phase=phase):
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

    dataset = PreprocessDataset(stores, config, train=train)
    data_loader = DataLoader(
        dataset,
        batch_size=config.preprocess.batch_size,
        num_workers=config.preprocess.num_workers,
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    # create output directory
    os.makedirs(output_path, exist_ok=True)

    # pre-process
    modify(config, data_loader, output_path, set=config.preprocess.set)


def get_data_keys(phase="train"):
    """ 
    Return the data keys depending on the phase.
    
    """

    data_keys = [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph"]

    if phase == "test":
        data_keys += [
                "labels",
                "label_columns",
                "track_ids",
                "seq_name"
            ]
    
    return data_keys

# Modified from https://github.com/uber-research/LaneGCN/blob/master/preprocess_data.py
def to_numpy(data):
    """
    Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data

# Modified from https://github.com/uber-research/LaneGCN/blob/master/preprocess_data.py
def to_int16(data):
    """
    Convert dict, list and arrays to int16.
    """

    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data

# Modified from https://github.com/uber-research/LaneGCN/blob/master/preprocess_data.py
def modify(config, data_loader, output_path, set):
    """ 
    Iterate through the preprocessing dataloader and modify the data into graph structure.
    """

    store = data_loader.dataset.split
    for _, data in enumerate(data_loader):
        data = [dict(x) for x in data]

        out = []
        for j in range(len(data)):
            out.append(preprocess(to_long(gpu(data[j])), config.dataset.cross_dist))            

        for j, graph in enumerate(out):
            idx = graph['idx']
            store[idx]['graph']['left'] = graph['left']
            store[idx]['graph']['right'] = graph['right']

    f = open(os.path.join(output_path, "processed_" + set + ".p"), 'wb')
    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

# Modified from https://github.com/uber-research/LaneGCN/blob/master/preprocess_data.py
class PreprocessDataset():
    def __init__(self, split, config, train=True):
        self.split = split
        self.config = config
        self.train = train

    def __getitem__(self, idx):
        from src.dataset import from_numpy, ref_copy

        data = self.split[idx]
        graph = dict()
        for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats']:
            graph[key] = ref_copy(data['graph'][key])
        graph['idx'] = idx
        return graph

    def __len__(self):
        return len(self.split)

# Modified from https://github.com/uber-research/LaneGCN/blob/master/preprocess_data.py
def preprocess(graph, cross_dist, cross_angle=None):
    """
    Pre-process the graph data.
    
    """
    
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    out = dict()
    out['left'] = left
    out['right'] = right
    out['idx'] = graph['idx']
    return out

# Modified from https://github.com/uber-research/LaneGCN/blob/master/preprocess_data.py
def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data
