"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Evaluation of anomaly detection performance.

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

import os
import sys
import numpy
import pickle
import pandas as pd
import math
import sklearn.metrics
import torch

import src.utils


class AnomalyEvaluator(object):
    
    def __init__(self, y_true, y_pred, y_true_sub=None) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_true_sub = y_true_sub

        self.metrics_list = ["AUPR-Abnormal", "AUPR-Normal", "AUROC", "FPR-95%-TPR"]

        self.eval_path = ""


    def evaluate(self, eval_path):

        # set evaluation path
        self.eval_path = eval_path

        # filter ignore regions
        self.filter_ignore_regions()
        
        # EVALUATION: Anomaly Detection

        # compute and show major anomaly metrics
        self.anomaly_metrics_major()
        _ = self.get_metrics(type="major")
        self.print_results(type="major")

        # EVALUATION: Anomaly Types

        # compute and show metrics
        self.anomaly_metrics_minor()
        _ = self.get_metrics(type="minor")
        self.print_results(type="minor")


    def export(self, results):

        # eval directory
        if not os.path.isdir(self.eval_path):
            os.makedirs(self.eval_path)

        # save results
        torch.save(results, os.path.join(self.eval_path, "results.pkl"))


    def log(self):
        log = os.path.join(self.eval_path, "log")
        
        sys.stdout = src.utils.Logger(log)


    def filter_ignore_regions(self):

        keep_index = self.y_true < 2.
        self.y_true = self.y_true[keep_index]
        self.y_pred = self.y_pred[keep_index]
        self.y_true_sub = self.y_true_sub[keep_index]


    def anomaly_metrics_major(self):

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.y_true, self.y_pred, pos_label=1, drop_intermediate=False)
        auroc = sklearn.metrics.roc_auc_score(self.y_true, self.y_pred)
        aupr_abnormal = sklearn.metrics.average_precision_score(self.y_true, self.y_pred)
        aupr_normal = sklearn.metrics.average_precision_score(1-self.y_true, -self.y_pred)
        fpr95tpr = fpr_at_95tpr(tpr, fpr)

        self.metrics_major = {"FPR": fpr, "TPR": tpr, "AUROC": auroc, 
        "AUPR-Abnormal": aupr_abnormal, "AUPR-Normal": aupr_normal, "FPR-95%-TPR": fpr95tpr}


    def anomaly_metrics_minor(self):

        # identify unique anomaly types
        anomaly_types = numpy.unique(self.y_true_sub)

        # remove void class, i.e. type == -1 (it is also used for the major class being normal)
        anomaly_types = anomaly_types[anomaly_types >= 0]
        
        # loop over anomaly types
        self.metrics_minor = {}

        for _, anomaly_code in enumerate(anomaly_types):

            # ignore all other types for evaluation
            y_t, y_tm, y_s = filter_anomalies(self.y_true, self.y_true_sub, self.y_pred, anomaly_code)

            # check if at least one positive class exists
            if not all_equal(y_t):
                
                # log
                anomaly_type = decode_anomaly_type(int(anomaly_code))

                # evaluate metric for each anomaly type
                metrics = compute_anomaly_metrics(y_t, y_s)

                # log metrics
                metrics["anomaly_code"] = anomaly_code
                self.metrics_minor[anomaly_type] = metrics


    def get_major_results_dict(self):

        major_result_dict = {k: self.metrics_major[k] for k in self.metrics_list}

        return major_result_dict


    def get_minor_results_dict(self):

        minor_result_dict = dict()

        for k in self.metrics_list:
            for clsmin, v in self.metrics_minor.items():
                minor_result_dict["{}/{}".format(k, clsmin)] = v[k] 

        return minor_result_dict


    def get_metrics(self, type="major"):

        if type == "major":
            return self.metrics_major
        elif type == "minor":
            return self.metrics_minor


    def print_results(self, type="major"):

        print("\n\nResults Summary {}\n\n".format(type.upper()))

        if type == "major":
            for k in ["AUPR-Abnormal", "AUPR-Normal", "AUROC", "FPR-95%-TPR"]:
                print("{:15}: {:8.2f}".format(k, round(self.metrics_major[k]*100, 2)))
        elif type == "minor":
            metric_keys = ["AUPR-Abnormal", "AUPR-Normal", "AUROC", "FPR-95%-TPR"]
            header = ["Anomaly Type"] + metric_keys
            out_str = "{:<20} {:<15} {:<15} {:<15} {:<15}"
            print(out_str.format(*header))
            for anomaly_type, metrics in self.metrics_minor.items():
                row = [anomaly_type]
                for mk in metric_keys:
                    row.append(round(metrics[mk]*100, 2))
                #print(row)
                print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(*row))


    def save_results(self, eval_dir):

        self.result_dict = dict()
        self.result_dict["ytrue"] = self.y_true
        self.result_dict["ypred"] = self.y_pred
        self.result_dict["ytruesub"] = self.y_true_sub

        with open(os.path.join(eval_dir, "results.pkl"), "wb") as fout:
            pickle.dump(self.result_dict, fout)
       

def compute_anomaly_metrics(anomaly_gts, anomaly_scores):

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(anomaly_gts, anomaly_scores, pos_label=1, drop_intermediate=False)
    auroc = sklearn.metrics.roc_auc_score(anomaly_gts, anomaly_scores)
    aupr_abnormal = sklearn.metrics.average_precision_score(anomaly_gts, anomaly_scores)
    aupr_normal = sklearn.metrics.average_precision_score(1-anomaly_gts, -anomaly_scores)
    fpr95tpr = fpr_at_95tpr(tpr, fpr)

    metrics = {"FPR": fpr, "TPR": tpr, "AUROC": auroc, 
    "AUPR-Abnormal": aupr_abnormal, "AUPR-Normal": aupr_normal, "FPR-95%-TPR": fpr95tpr}

    return metrics


def fpr_at_95tpr(tpr, fpr):
    hit = False
    tpr_95_lb = 0
    tpr_95_ub = 0
    fpr_95_lb = 0
    fpr_95_ub = 0

    for i in range(len(tpr)):
        if tpr[i] > 0.95 and not hit:
            tpr_95_lb = tpr[i - 1]
            tpr_95_ub = tpr[i]
            fpr_95_lb = fpr[i - 1]
            fpr_95_ub = fpr[i]
            hit = True

    s = pd.Series([fpr_95_lb, numpy.nan, fpr_95_ub], [tpr_95_lb, 0.95, tpr_95_ub])

    s = s.interpolate(method="index")

    return s.iloc[1]


def compute_fde(forecasted_trajectory: numpy.ndarray, gt_trajectory: numpy.ndarray) -> float:
    """Compute Final Displacement Error for reconstruction.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        fde: Final Displacement Error

    """
    fde = math.sqrt(
        (forecasted_trajectory[0, 0] - gt_trajectory[0, 0]) ** 2
        + (forecasted_trajectory[0, 1] - gt_trajectory[0, 1]) ** 2
    )
    return fde


def decode_anomaly_type(anomaly_code):
    """
    Possible anomaly types, not all of them occur in the r-u-maad dataset.

    """
    
    anomaly_dict =     {0: 'void',
                        1: 'aggr. overtaking (l)',
                        2: 'aggr. overtaking (r)',
                        3: 'aggr. shearing (l)',
                        4: 'aggr. shearing (r)',
                        5: 'backwards',
                        6: 'cancel turn',
                        7: 'drifting',
                        8: 'enter wrong lane',
                        9: 'ghost driver',
                        10: 'last min. turn',
                        11: 'leave road',
                        12: 'prevent merge',
                        13: 'thwarting',
                        14: 'pushing away',
                        15: 'red light running',
                        16: 'skidding',
                        17: 'staggering',
                        18: 'swerving (l)',
                        19: 'swerving (r)',
                        20: 'tailgating',
                        21: 'take right-of-way',
                        22: 'thwarting',
                        23: 'other',
                        24: 'ignore region',
                        25: 'speeding',
                        26: 'stopping'}

    return anomaly_dict[anomaly_code]


def decode_normal_type(normal_code):
    """
    Possible normal driving types, not all of them occur in the r-u-maad dataset.
    """

    normalies = ["void",
                "accelerate",
                "brake",
                "following",
                "lane change (l)",
                "lane change (r)",
                "merge to (l)",
                "merge to (r)",
                "multi-overtake (l)",
                "multi-overtake (r)",
                "overtaking (l)",
                "overtaking (r)",
                "side-by-side",
                "straight",
                "turn (l)",
                "turn (r)",
                "other"]

    decode_dict = {i: a for i, a in enumerate(normalies)}

    return decode_dict[normal_code]

def filter_anomalies(y_true_major,y_true_minor, y_score, anomaly):

    # convert lists to numpy arrays
    y_true_major = numpy.array(y_true_major)
    y_true_minor = numpy.array(y_true_minor)
    y_score = numpy.array(y_score)

    # filter other anomalies
    valid_indices0 = numpy.where(y_true_major == 0)[0]
    valid_indices1 = numpy.where(y_true_major == 1)[0]
    valid_indices2 = numpy.where(y_true_minor == anomaly)[0]
    valid_indices3 = numpy.intersect1d(valid_indices1, valid_indices2)
    valid_indices = numpy.union1d(valid_indices3, valid_indices0)

    y_true_minor = y_true_minor[valid_indices]
    y_true_major = y_true_major[valid_indices]
    y_score = y_score[valid_indices]

    return y_true_major, y_true_minor, y_score

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)