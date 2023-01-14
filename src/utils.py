"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Different utility functions used throughout the project.

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

import numpy
import sys
import os
import time
import datetime
import math
import networkx
import numpy as np
import random
import torch
from torch import optim

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")


class Logger(object):
    def __init__(self, log):
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(trajs, trajs_rel, norm_lap_matr=True, adj_type="relative"):
    """
    Define graph structure and compute the adjacency matrix given by adj_type.
    
    Parameters
    ------------
    trajs: array
        Actor trajectories in absolute coordinates [NxTx2]
    trajs_rel: array
        Relative actor trajectories [NxTx2]
    
    Returns
    ------------
    V: tensor
        Tensor of node features, relative trajectories [T, N, 2]
    A: tensor
        Normalized Laplacian matrices  [T, N, N]

    """

    N = trajs.shape[0]
    T = trajs.shape[1]

    V = numpy.zeros((T, N, 2))
    A = numpy.zeros((T, N, N))
    for t in range(T):
        step = trajs[:, t, :]
        step_rel = trajs_rel[:, t, :]
        for h in range(len(step)):
            V[t, h, :] = step_rel[h]
            A[t, h, h] = 1  # self-aggregation
            for k in range(h + 1, len(step)):
                if adj_type == "relative":
                    l2_norm = anorm(step_rel[h], step_rel[k])
                elif adj_type == "absolute":
                    l2_norm = anorm(step[h], step[k])
                elif adj_type == "identity":
                    l2_norm = 0
                else:
                    sys.exit("Unkonwn adjacency type {}, choose from 'relative' or 'absolute'.".format(adj_type))
                A[t, h, k] = l2_norm
                A[t, k, h] = l2_norm
        if norm_lap_matr:  # normalize laplacian matrix
            G = networkx.from_numpy_matrix(A[t, :, :])
            A[t, :, :] = networkx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)


class Optimizer(object):
    def __init__(self, params, config, coef=None):
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = [params]

        if coef is None:
            coef = [1.0] * len(params)
        else:
            if isinstance(coef, list) or isinstance(coef, tuple):
                assert len(coef) == len(params)
            else:
                coef = [coef] * len(params)
        self.coef = coef

        param_groups = []
        for param in params:
            param_groups.append({"params": param, "lr": 0})

        self.opt = optim.Adam(param_groups, weight_decay=0)

        self.lr_func = StepLR(config.training.lr, config.training.lr_epochs)

        self.clip_grads = False


    def zero_grad(self):
        self.opt.zero_grad()


    def step(self, epoch):
        if self.clip_grads:
            self.clip()

        lr = self.lr_func(epoch)
        for i, param_group in enumerate(self.opt.param_groups):
            param_group["lr"] = lr * self.coef[i]
        self.opt.step()
        return lr


    def clip(self):
        low, high = self.clip_low, self.clip_high
        params = []
        for param_group in self.opt.param_groups:
            params += list(filter(lambda p: p.grad is not None, param_group["params"]))
        for p in params:
            mask = p.grad.data < low
            p.grad.data[mask] = low
            mask = p.grad.data > high
            p.grad.data[mask] = high


    def load_state_dict(self, opt_state):
        self.opt.load_state_dict(opt_state)


class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.lr = lr
        self.lr_epochs = lr_epochs

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        return self.lr[idx]
