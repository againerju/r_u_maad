"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Preprocess the data(csv), build graph from the HDMAP and saved as pkl.

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

import argparse
import os
import hydra

from src.preprocess import sliding_window_preprocess, pickle_preprocess
from experiment import update_config_for_preprocessing

os.umask(0)

if __name__ == "__main__":

    # we use a plain argparser to parse everything that we need outside of what is being managed
    # by hydra (e.g. everything necessary to set up hydra)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="preprocess_data.yaml"
    )

    # we treat all the left over arguments as overrides to hydra experiment config nodes
    job_args, config_overrides = parser.parse_known_args()

    # use hydra compose API to maintain easier compatatibility with torch DDP
    # overrides need to be passed as "node.subnode=value", e.g. taining.batch_size=16
    with hydra.initialize(config_path="config", job_name="data_preprocessing"):
        config = hydra.compose(config_name=job_args.config,
                            overrides=config_overrides)

    # update config for preprocessing
    config = update_config_for_preprocessing(config)

    # paths
    data_root = os.path.join(os.getcwd(), "datasets", "r_u_maad")
    sw_data_path = os.path.join(data_root, "sliding_window")
    process_data_path = os.path.join(data_root, "preprocess")
    
    # select set
    if config.preprocess.set == "train":
        set_name = config.dataset.train_split
    elif config.preprocess.set == "val":
        set_name = config.dataset.val_split
    elif config.preprocess.set == "test":
        set_name = config.dataset.test_split

    input_path = os.path.join(data_root, set_name, "data")
    sw_path = os.path.join(sw_data_path, set_name)
    process_path = process_data_path

    # sliding window
    sliding_window_preprocess(input_path, sw_path, config.preprocess.n_cpus, n_samples=config.preprocess.n_samples, size=config.dataset.seq_len, stride=config.preprocess.stride)

    # pre-processing
    pickle_preprocess(config, sw_path, process_path, phase=config.preprocess.set)
