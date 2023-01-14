"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Linear Temporal Interpolation (LTI) model definition.

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

import numpy
from typing import Dict, List
from torch import Tensor, nn
from argoverse.evaluation.eval_forecasting import get_ade


class LTI(nn.Module):

    def __init__(self):
        super(LTI, self).__init__()


    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:

        results = dict()
        results["reg"] = []
        results["gt_preds"] = []

        for b, trajs in enumerate(data["gt_preds"]):

            approx_trajs = []

            for traj in trajs:
                # trajectory features
                start_pos = traj[0, :]
                end_pos = traj[-1, :]
                n_ts = traj.shape[0]

                # linear temporal approximation
                x_interp = numpy.linspace(start_pos[0], end_pos[0], n_ts)
                y_interp = numpy.linspace(start_pos[1], end_pos[1], n_ts)
                approx_traj = numpy.zeros(traj.shape)
                approx_traj[:, 0] = x_interp
                approx_traj[:, 1] = y_interp

                approx_trajs.append(approx_traj)
   
            approx_trajs = numpy.stack(approx_trajs)[:, numpy.newaxis]

            results["reg"].append(approx_trajs)
            results["gt_preds"].append(trajs)

        return results

    def score_anomaly(self, out):
        scores = []

        # select target node for which we want to score anomalies
        # r-u-maad: agent with id=0
        target_node_id = 0
        
        for i in range(len(out["reg"])):

            x_trgt = out["gt_preds"][i][target_node_id]
            x_pred = out["reg"][i][target_node_id, 0]

            score = get_ade(x_pred, x_trgt)

            scores.append(score)
        
        return scores
