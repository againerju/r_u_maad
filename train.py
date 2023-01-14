"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Training script.

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

import os
import hydra
import argparse
import numpy as np
import time
import pickle
from tqdm import tqdm
import torch

from src.utils import seed_all
from experiment import init_experiment, create_experiment_dir, init_log, log_files, update_config
from experiment import run_with_config, add_experiment_dir, get_dataloader

os.umask(0)

def main():

    # we use a plain argparser to parse everything that we need outside of what is being managed
    # by hydra (e.g. everything necessary to set up hydra)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="default.yaml"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="r_u_maad_2022"
    )

    # we treat all the left over arguments as overrides to hydra experiment config nodes
    job_args, config_overrides = parser.parse_known_args()

    # use hydra compose API to maintain easier compatatibility with torch DDP
    # overrides need to be passed as "node.subnode=value", e.g. taining.batch_size=16
    with hydra.initialize(config_path="config", job_name=job_args.job_name):
        config = hydra.compose(config_name=job_args.config,
                            overrides=config_overrides)

    # seed all
    seed_all(config.misc.seed)

    # experiment tracking
    experiment_dir = create_experiment_dir(config)
    init_log(experiment_dir)
    log_files(experiment_dir)
    config = add_experiment_dir(config, experiment_dir)

    # update configuration with static parameters
    config = update_config(config)

    # initialize experiment
    run_with_config(job_args.config)
    config, Dataset, collate_fn, model, loss, post_process, opt = init_experiment(config)
    
    # data loader for training
    dataset = Dataset(config.dataset.train_split, config, train=True)
    train_loader = get_dataloader(config, dataset, collate_fn, phase="train")

    # data loader for evaluation
    dataset = Dataset(config.dataset.val_split, config, train=False)
    val_loader = get_dataloader(config, dataset, collate_fn, phase="val")

    # train
    epoch = 0
    remaining_epochs = int(np.ceil(config.training.num_epochs - epoch))
    print("Start Training...")
    for i in range(remaining_epochs):
        train(epoch + i, config, experiment_dir, train_loader, model, loss, post_process, opt, val_loader)


def train(epoch, config, exp_dir, train_loader, net, loss, post_process, opt, val_loader=None):

    net.train()

    # parameters
    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config.training.save_freq * num_batches))
    display_iters = int(
        config.training.display_iters / (config.training.batch_size)
    )
    val_iters = int(config.training.val_iters / (config.training.batch_size))

    start_time = time.time()
    metrics = dict()

    # iterate through training data
    for i, data in enumerate(tqdm(train_loader)):
        epoch += epoch_per_batch
        data = dict(data)

        # forward
        output = net(data)

        # loss
        loss_out = loss(output, data, epoch)
        post_out = post_process(output, data)
        post_process.append(metrics, loss_out, post_out)

        # backward
        opt.zero_grad()
        loss_out["loss"].backward()
        lr = opt.step(epoch)

        num_iters = int(np.round(epoch * num_batches))
        if True and (
            num_iters % save_iters == 0 or epoch >= config.training.num_epochs
        ):
            save_ckpt(net, opt, exp_dir, epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            if True:
                curr_losses = post_process.display(metrics, dt, epoch, lr)

            # save train loss
            losses = {}
            loss_file = os.path.join(exp_dir, "train_loss.p")
            if os.path.exists(loss_file):
                with open(loss_file,'rb') as file:
                    losses = pickle.load(file)
            
            losses[epoch] = curr_losses
            with open(loss_file,'wb') as file:
                pickle.dump(losses, file)

            start_time = time.time()
            metrics = dict()

        if num_iters % val_iters == 0:
            val(exp_dir, val_loader, net, loss, post_process, epoch)

        if epoch >= config.training.num_epochs:
            val(exp_dir, val_loader, net, loss, post_process, epoch)
            return


def val(exp_dir, data_loader, net, loss, post_process, epoch):
    net.eval()

    start_time = time.time()
    metrics = dict()
    
    # iterate through validation data
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            # predict
            output = net(data)
            # loss
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    if True:
        curr_losses = post_process.display(metrics, dt, epoch)
    
    losses = {}
    loss_file = os.path.join(exp_dir, "val_loss.p")
    if os.path.exists(loss_file):
        with open(loss_file,'rb') as file:
            losses = pickle.load(file)
    
    losses[epoch] = curr_losses
    with open(loss_file,'wb') as file:
        pickle.dump(losses, file)

    net.train()


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )


if __name__ == "__main__":
    main()
