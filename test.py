"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Test script.

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

import argparse
import os
import hydra
from tqdm import tqdm
import numpy
import torch
from torch.utils.data import DataLoader

from src.dataset import ArgoTestDataset
from src.utils import load_pretrain, get_timestamp
import src.evaluation
from experiment import *

os.umask(0)

def main():

    # CONFIG

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
    
    # update configuration with static parameters
    config = update_config(config, phase="test")

    # EXPERIMENT

    # experiment tracking
    experiment_dir = get_experiment_dir(config)
    eval_dir = create_eval_dir(experiment_dir)
    init_log(eval_dir)
    log_files(eval_dir)
    config = add_experiment_dir(config, experiment_dir)

    # init experiment
    run_with_config(job_args.config)
    config, collate_fn, model, loss = init_experiment(config, phase="test")

    # data loader for evaluation
    dataset = ArgoTestDataset(config.dataset.test_split, config)
    data_loader = DataLoader(
        dataset,
        batch_size=config.testing.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    # MODEL
    is_linear_model = True
    if config.model.name in ["seq2seq", "stgae", "lanegcn_ae"]:
            
        # load pretrain model
        ckpt_path = get_ckpt_path(experiment_dir, ckpt_file=config.testing.eval_ckpt)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(model, ckpt["state_dict"])
        model.eval()
        is_linear_model = False

    # INFERNCE

    # begin inference
    res = dict()
    for k in ["gts", "labels", "label_columns", "score"]:
        res[k] = dict()

    for _, data in enumerate(tqdm(data_loader)):
        data = dict(data)
        with torch.no_grad():

            # predict
            output = model(data)

            # compute anomaly score 
            if not is_linear_model:
                scores = loss.score_anomaly(output, data)
            else:
                scores = model.score_anomaly(output)

            results = [x[:, 0] for x in output["reg"]]
            output["reg"] = results

            for i, argo_id in enumerate(data["argo_id"]):

                res["gts"][argo_id] = data["gt_preds"][i].numpy() if "gt_preds" in data else None
                res["labels"][argo_id] = data["labels"][i].numpy()
                res["label_columns"][argo_id] = data["label_columns"][i]
                res["score"][argo_id] = scores[i]

    # EVALUATION
    y_true = []
    y_true_sub = []
    y_pred = []

    for argo_id in res["gts"].keys():
    
        # score
        score = res["score"][argo_id]

        # labels
        validity_code = res['labels'][argo_id][0, 0]
        labels = res["labels"][argo_id][0, 1::][numpy.newaxis, :]
        if validity_code != 1: labels[0, 0] = 2.0
        maj_label = labels[0, 0]
        min_label = labels[0, 1]

        # append
        y_true.append(maj_label)
        y_true_sub.append(min_label)
        y_pred.append(score)

    y_true = numpy.array(y_true)
    y_true_sub = numpy.array(y_true_sub)
    y_pred = numpy.array(y_pred)

    # initialize evaluator
    evaluator = src.evaluation.AnomalyEvaluator(y_true, y_pred, y_true_sub=y_true_sub)

    # evaluate
    evaluator.evaluate(eval_path=eval_dir)

    # export results
    evaluator.export(results)

    # Create evaluation log
    evaluator.log()


if __name__ == "__main__":
    main()
