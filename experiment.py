"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Functions for experiment management.

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

from operator import mod
import sys
import os
import datetime
import shutil
import glob
import pickle
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import  DataLoader

from src import utils


def get_data_root():

    proj_root = os.path.join(os.getcwd())
    data_root = os.path.join(proj_root, "datasets", "r_u_maad")

    return data_root

def update_config(cfg, phase="train"):

    data_root = get_data_root()

    if phase == "train":
        cfg.dataset.train_split = os.path.join(data_root, "sliding_window", cfg.dataset.train_split)
        cfg.dataset.val_split = os.path.join(data_root, "sliding_window", cfg.dataset.val_split)

        cfg.dataset.preprocess_train = os.path.join(data_root, "preprocess", cfg.dataset.preprocess_train)
        cfg.dataset.preprocess_val = os.path.join(data_root, "preprocess", cfg.dataset.preprocess_val)
    
    else:
        cfg.dataset.test_split = os.path.join(data_root, "sliding_window", cfg.dataset.test_split)
        cfg.dataset.preprocess_test = os.path.join(data_root, "preprocess", cfg.dataset.preprocess_test)

    return cfg

def update_config_for_preprocessing(cfg):

    cfg.dataset.preprocess = False
    cfg.dataset.cross_angle = 0.5 * np.pi
    if cfg.preprocess.set in ["train", "val"]:
        cfg.dataset.include_labels = False
    else:
        cfg.dataset.include_labels = True

    print("#--- CONFIG ---#")
    print(OmegaConf.to_yaml(cfg))

    return cfg

def add_experiment_dir(cfg, experiment_dir):
    cfg.experiment.path = experiment_dir
    return cfg


def run_with_config(config_file):
    print("Running experiment with config {}...".format(os.path.join(".", "config", config_file)))


def get_dataloader(cfg, dataset, collate_fn, phase="train"):

    drop_last = False
    if phase == "train":
        drop_last = True

    if phase == "train":
        batch_size = cfg.training.batch_size
    elif phase == "val":
        batch_size = cfg.training.val_batch_size
    elif phase == "test":
        batch_size = cfg.testing.batch_size 
    else:
        raise ValueError

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=drop_last,
    )


def init_experiment(cfg, phase="train"):

    print("Initialize experiment...")

    print("#--- CONFIG ---#")
    print(OmegaConf.to_yaml(cfg))

    # data
    from src.dataset import ArgoDataset, collate_fn

    # post-process
    from src.postprocess import PostProcess

    # model and loss
    if cfg.model.name == "cvm":
        from models.cvm import CVM
        model = CVM()
        from src.losses import RegressionLoss
        loss = RegressionLoss(cfg)
    elif cfg.model.name == "lti":
        from models.lti import LTI
        model = LTI()
        from src.losses import RegressionLoss
        loss = RegressionLoss(cfg)
    elif cfg.model.name == "seq2seq":
        from models.seq2seq import Seq2SeqNet
        from src.losses import RegressionLoss
        model = Seq2SeqNet(cfg)
        loss = RegressionLoss(cfg)
    elif cfg.model.name == "stgae":
        from models.stgae import STGAE
        from src.losses import RegressionLoss
        model = STGAE(cfg)
        loss = RegressionLoss(cfg)
    elif cfg.model.name == "lanegcn_ae":
        from models.lanegcn_ae import LaneGCNAE
        from src.losses import RegressionLoss
        model = LaneGCNAE(cfg)
        loss = RegressionLoss(cfg)
    else:
        raise ValueError(f"Unknown model {cfg.model.name}.")

    post_process = PostProcess(cfg)

    model.cuda()
    loss.cuda()
    post_process.cuda()

    # optimizer
    params = model.parameters()
    optimizer = utils.Optimizer(params, cfg)

    if phase == "train":
        return cfg, ArgoDataset, collate_fn, model, loss, post_process, optimizer
    elif phase == "test":
        return cfg, collate_fn, model, loss
    else:
        raise ValueError(f"Invalid phase {phase}")


def get_date_and_time_string():

    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    return date_time


def create_experiment_dir(cfg):

    # experiment name
    experiment_name = get_date_and_time_string() + '_' + cfg.model.name

    # experiment root
    experiment_root = os.path.join(os.getcwd(), 'experiments')

    # experiment directory
    experiment_dir = os.path.join(experiment_root, experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)

    return experiment_dir


def get_experiment_dir(cfg):

    return os.path.join(os.getcwd(), 'experiments', cfg.testing.runs[cfg.testing.run])


def create_eval_dir(experiment_dir):

    eval_dir = os.path.join(experiment_dir, "test_{}".format(get_date_and_time_string()))

    os.makedirs(eval_dir, exist_ok=True)

    return eval_dir


def init_log(experiment_dir):

    log = os.path.join(experiment_dir, "log")

    sys.stdout = utils.Logger(log)


def log_files(experiment_dir):

    proj_dir = os.getcwd()

    for dir in [".", "models", "src", "config"]:
        src_dir = os.path.join(proj_dir, dir)
        dst_dir = os.path.join(experiment_dir, "files", dir)
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if f.endswith(".py"):
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

def get_ckpt_path(experiment_dir, ckpt_file=None):
    """
    If ckpt_file is given, try to load checkpoint from ckpt_file, 
    else load the val_loss.p file and select the checkpoint with the lowest validation loss.
    
    """

    if ckpt_file:

        ckpt_path = os.path.join(experiment_dir, ckpt_file)
        
        try:
            os.path.exists(ckpt_path)
        except:
            print("Checkpoint {} does not exist.".format(ckpt_file))
    
    else:

        ckpt_paths = [p for p in sorted(glob.glob(os.path.join(experiment_dir, "*.ckpt")))]

        val_loss_log = "val_loss.p"

        try:
            with open(os.path.join(experiment_dir, val_loss_log),'rb') as file:
                losses = pickle.load(file)
        except:
            print("Log of validation loss, {},  missing. Please specify a checkpoint file.".format(val_loss_log))
        
        checkpoint_numbers = [Path(x).stem for x in ckpt_paths]
        lowest_loss_epoch = min(losses.items(), key=lambda x: float(x[1]["loss"]))[0]

        closest_checkpoint_number = min(range(len(checkpoint_numbers)), key=lambda i: abs(float(checkpoint_numbers[i])-lowest_loss_epoch))

        ckpt_path = ckpt_paths[closest_checkpoint_number]

    print("Checkpoint {} selected.\n".format(os.path.split(ckpt_path)[1]))

    return ckpt_path