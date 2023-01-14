# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified; date of modification: 14/01/2023

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap


class ArgoDataset(Dataset):

    def __init__(self, split, config, train=True, seq_len=16):
        self.config = config
        self.train = train
        self.seq_len = seq_len
        self.pred_range = [-100, +100, -100, +100]
        
        if 'preprocess' in config.dataset.keys() and config.dataset.preprocess:
            if train:
                self.split = np.load(self.config.dataset.preprocess_train, allow_pickle=True)
            else:
                self.split = np.load(self.config.dataset.preprocess_val, allow_pickle=True)
        else:
            self.avl = ArgoverseForecastingLoader(split)
            self.avl.seq_list = sorted(self.avl.seq_list)
            self.am = ArgoverseMap()

        self.set_label_columns()
            
    def __getitem__(self, idx):

        if self.config.dataset.preprocess:
            data = self.split[idx]

            # add empty values to data dictionary
            data["obs_traj"] = []
            data["obs_traj_rel"] = []
            data["v_obs"] = []
            data["A_obs"] = []

            new_data = dict()
            for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph', "min_agent_dist", "min_agent_vel", "min_centerline_dist", 
                        'obs_traj', 'obs_traj_rel', 'v_obs', 'A_obs']:
                if key in data:
                    new_data[key] = ref_copy(data[key])
            data = new_data

            return data

        # Read raw data
        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['idx'] = idx

        if 'raster' in self.config and self.config['raster']:
            x_min, x_max, y_min, y_max = self.pred_range
            cx, cy = data['orig']

            region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
            raster = self.map_query.query(region, data['theta'], data['city'])

            data['raster'] = raster
            return data

        if "build_lane_graph" in self.config and not self.config["build_lane_graph"]:
            data['graph'] = None
        else:
            data['graph'] = self.get_lane_graph(data)
        return data
    
    def __len__(self):
        if 'preprocess' in self.config.dataset and self.config.dataset.preprocess:
            return len(self.split)
        else:
            return len(self.avl)


    def set_label_columns(self):

        self.label_columns = ["VALID", "BEHAVIOR", "SUBCLASS"]


    def read_argo_data(self, idx):
        city = copy.deepcopy(self.avl[idx].city)

        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        df = copy.deepcopy(self.avl[idx].seq_df)
        
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)

        if self.config.dataset.include_labels:
            column_headers = list(df.columns.values)

            # if label columns are not available, e.g. for train or val data, 
            # set all time steps valid and the labels to zero
            # VALID = 1, all others = 0
            if not set(self.label_columns).issubset(column_headers):
                for i, column_name in enumerate(self.label_columns):
                    if column_name == "VALID":
                        df[column_name] = 1
                    else:
                        df[column_name] = 0

            # extract labels given the label columns
            labels = [df[x].to_numpy().reshape(-1, 1) for x in self.label_columns]
            labels = np.concatenate(labels, 1)

        # get all time steps
        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        # group df by track_id and object_type
        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        if self.config.dataset.include_labels:
            # if true, the controlled vehicle is used as agent
            agt_idx = obj_type.index('CONTROLLED')
        else:
            # else, the agent is used
            agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]
       
        agt_id = keys[agt_idx][0]
        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        if self.config.dataset.include_labels:
            curr_agent_labels = labels[idcs]
            agt_label = np.empty(len(self.label_columns))
            agt_label.fill(np.NaN)

            if (self.seq_len-1) in steps[idcs]:
                agt_label = curr_agent_labels[-1]

        del keys[agt_idx]
        ctx_trajs, ctx_steps, ctx_labels, ctx_ids = [], [], [], []
        for key in keys:
            idcs = objs[key]
            ctx_ids.append(key[0])
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])
            if self.config.dataset.include_labels:
                curr_agent_labels = labels[idcs]
                agt_label_ctx = np.empty(len(self.label_columns))
                agt_label_ctx.fill(np.NaN)

                if 15 in steps[idcs]:
                    agt_label_ctx = curr_agent_labels[-1]
                
                ctx_labels.append(agt_label_ctx)

        data = dict()
        data['city'] = city
        data['seq_name'] = os.path.splitext(self.avl.seq_list[idx].name)[0]
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        data['track_id'] = [agt_id] + ctx_ids
        if self.config.dataset.include_labels:
            data["labels"] = [agt_label] + ctx_labels
            data["label_columns"] = self.label_columns
        else:
            # Create dummy labels, that will not be used further
            data["labels"] = [None for i in range(len(data["trajs"]))]
            data["label_columns"] = self.label_columns
        
        return data
    
    def get_obj_feats(self, data):
        orig = data['trajs'][0][self.seq_len-1].copy().astype(np.float32)

        if self.train and self.config.dataset.rot_aug:
            theta = np.random.rand() * np.pi * 2.0
        else:
            pre = data['trajs'][0][self.seq_len-2] - orig
            theta = np.pi - np.arctan2(pre[1], pre[0])

        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, ctrs, gt_preds, has_preds, labels, track_ids = [], [], [], [], [], []
        for traj, step, label, tid in zip(data['trajs'], data['steps'], data["labels"], data["track_id"]):
            if (self.seq_len-1) not in step:
                continue

            gt_pred = np.zeros((self.seq_len, 2), np.float32)
            has_pred = np.zeros(self.seq_len, np.bool)
            future_mask = np.logical_and(step >= 0, step < self.seq_len)
            post_step = step[future_mask] - 0
            post_traj = traj[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = 1
            
            obs_mask = step < self.seq_len
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]
            
            for i in range(len(step)):
                if step[i] == (self.seq_len - 1) - (len(step) - 1) + i:
                    break
            step = step[i:]
            traj = traj[i:]

            feat = np.zeros((self.seq_len, 3), np.float32)
            feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat[step, 2] = 1.0

            x_min, x_max, y_min, y_max = self.pred_range
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            ctrs.append(feat[-1, :2].copy())
            feat[1:, :2] -= feat[:-1, :2]
            feat[step[0], :2] = 0
            feats.append(feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

            labels.append(label)

            track_ids.append(tid)

        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        labels = np.asarray(labels, np.float32)

        data['feats'] = feats
        data['ctrs'] = ctrs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds
        data['track_ids'] = track_ids

        if self.config.dataset.include_labels:
            data["labels"] = labels
        else:
            # Remove dummy labels
            del data["labels"]
            del data["label_columns"]

        return data

 
    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = self.pred_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)
        
        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane
            
        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1
            
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))
            
            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))
            
        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count
        
        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]
            idcs = node_idcs[i]
            
            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if lane.predecessors is not None:
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])
                    
            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)
                    
        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        
        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)
        
        for key in ['pre', 'suc']:
            graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], self.config.preprocess.num_scales)
        return graph


class ArgoTestDataset(ArgoDataset):

    def __init__(self, split, config, train=False):

        self.config = config
        self.train = train
        split2 = config.dataset.test_split
        split = self.config.dataset.preprocess_test

        self.avl = ArgoverseForecastingLoader(split2)
        self.avl.seq_list = sorted(self.avl.seq_list)
        if self.config.dataset.preprocess:
            self.split = np.load(split, allow_pickle=True)
            

    def __getitem__(self, idx):
        if self.config.dataset.preprocess:
            data = self.split[idx]
            name_filter = filter(str.isdigit, self.avl.seq_list[idx].name[:-4])
            name_filtered = "".join(name_filter)
            data['argo_id'] = int(name_filtered) #160547

            new_data = dict()

            if self.config.dataset.include_labels:
                key_list = ['orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph','argo_id','city','labels','label_columns']
            else:
                key_list = ['orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph','argo_id','city']

            for key in key_list:
                if key in data:
                    new_data[key] = ref_copy(data[key])
            data = new_data
            
            return data

        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        name_filter = filter(str.isdigit, self.avl.seq_list[idx].name[:-4])
        name_filtered = "".join(name_filter)
        data['argo_id'] = int(name_filtered) #160547
        if "build_lane_graph" in self.config and not self.config["build_lane_graph"]:
            data['graph'] = None
        else:
            data['graph'] = self.get_lane_graph(data)
        data['idx'] = idx
        return data
    
    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

    
def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs

 
def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch
