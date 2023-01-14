# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified; date of modification: 14/01/2023

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import numpy
from numpy import ndarray


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out,data):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """
        Every display-iters print training/val information
        """

        if lr is not None:
            print("\nEpoch %3.3f\tlr %.5f\ttime %3.2f" % (epoch, lr, dt))
        else:
            print(
                "\n************************* Validation, time %3.2f *************************"
                % dt
            )

        display_str = ""
        out_dict = dict()
        
        # losses
        if "reg_loss" in metrics.keys():
            reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
            display_str += "reg {:4.2f}\t".format(reg)
            out_dict["reg_loss"] = reg

        loss = metrics["loss"] / (metrics["num_batch"] + 1e-10)

        # metrics
        preds = numpy.concatenate(metrics["preds"], 0)
        gt_preds = numpy.concatenate(metrics["gt_preds"], 0)
        has_preds = numpy.concatenate(metrics["has_preds"], 0)
        ade, fde = recon_metrics(preds, gt_preds, has_preds)
        out_dict["ade"] = ade
        out_dict["fde"] = fde

        display_str = "total_loss {:4.2f}\t".format(loss) + display_str
        display_str += "ade {:4.2f}\t".format(ade)
        display_str += "fde {:4.2f}\t".format(fde)
        display_str += "\n\n"

        print(display_str)

        return out_dict


def recon_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = numpy.asarray(preds, numpy.float32)
    gt_preds = numpy.asarray(gt_preds, numpy.float32)

    """batch_size x num_mods x num_preds"""
    err = numpy.sqrt(((preds - numpy.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade = err[:, 0].mean()
    fde = err[:, 0, 0].mean() # first element, since trajectories are reconstructed from present to past
  
    return ade, fde