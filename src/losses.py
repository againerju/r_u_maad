# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified; date of modification: 14/01/2023

import torch
from torch import Tensor, nn
from typing import Dict, List, Union

from argoverse.evaluation.eval_forecasting import get_ade
from src.utils import gpu


class PredLoss(nn.Module):

    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")


    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        reg = out["reg"]
        reg = torch.cat([x[0:1] for x in reg], 0)
        gt_preds = torch.cat([x[0:1] for x in gt_preds], 0)
        has_preds = torch.cat([x[0:1] for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (reg.sum())
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.config.model.num_mods, self.config.model.num_preds
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        reg = reg[row_idcs, min_idcs]
        loss_out["reg_loss"] += self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out


class RegressionLoss(nn.Module):

    def __init__(self, config):
        super(RegressionLoss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)


    def forward(self, out: Dict, data: Dict, epoch=-1) -> Dict:
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        loss_out["num_batch"] = 1
        return loss_out


    def score_anomaly(self, out, data):
        scores = []

        # select target node for which we want to score anomalies
        # r-u-maad: agent with id=0
        target_node_id = 0
        
        for i in range(len(out["reg"])):

            x_trgt = out["gt_preds"][i][target_node_id]
            x_pred = out["reg"][i][target_node_id, 0].cpu()

            score = get_ade(x_pred, x_trgt)

            scores.append(score)
        
        return scores
