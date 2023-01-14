"""
A Benchmark for Unsupervised Anomaly Detection in Multi-Agent Trajectories

Constant Velocity Model (CVM) definition.

See the License for the specific language governing permissions and
limitations under the License.

Written by Julian Wiederer, Julian Schmidt (2022)
"""

import numpy
from typing import Dict, List
from torch import Tensor, nn
from argoverse.evaluation.eval_forecasting import get_ade

class CVM(nn.Module):

    def __init__(self):
        super(CVM, self).__init__()


    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        
        results = dict()
        results["reg"] = []
        results["gt_preds"] = []

        for trajs in data["gt_preds"]:

            approx_trajs = []

            for traj in trajs:
                # constant velocity approximation
                start_pos = traj[0, :]
                velocity = traj[1, :] - traj[0, :]
                approx_traj = numpy.add(numpy.zeros(traj.shape), velocity[numpy.newaxis, :])
                approx_traj[0, :] = start_pos
                approx_traj = numpy.cumsum(approx_traj, axis=0)

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
