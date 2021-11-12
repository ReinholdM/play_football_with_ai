# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict

from malib.algorithm.common.trainer import Trainer
from malib.algorithm.mappo.loss import MAPPOLoss
from malib.backend.datapool.offline_dataset_server import Episode
import torch


def get_part_data_from_batch(batch_data, idx):
    # XXX(ziyu): suppose the first dimension is batch_size
    res = {}
    for k, v in batch_data.items():
        res[k] = v[idx]
    return res


class MAPPOTrainer(Trainer):
    def __init__(self, tid):
        super(MAPPOTrainer, self).__init__(tid)
        self._loss = MAPPOLoss()
        self.optimize_policy = True

    def optimize(self, batch):

        ppo_epoch = self.policy.custom_config["ppo_epoch"]
        for _ in range(ppo_epoch):
            if self.optimize_policy:
                total_results = self.update_policy(batch)
                # Todo (Linghui): rollout based on the updated policy to generate the new_batch to update shaping function
            else:
                total_results = self.update_shaping_function(
                    batch)  # Todo (Linghui): batch here should be new_batch collected from the updated policy
                # Fixme (Linghui): we need obs_batch, action_batch, and return_batch in the new_batch above

        return total_results

    def switch_optimization(self):
        self.optimize_policy = not self.optimize_policy
        """
            clean the gradient  of the theta w.r.t. phi
            if now it is to optimize the policy again
        """
        # if self.optimize_policy:
        #     self.grad_theta_wrt_phi_aggr = None
        #     self.grad_aggr_num = 0
        # else:
        #     self.grad_theta_wrt_phi_aggr = np.divide(self.grad_theta_wrt_phi_aggr, self._loss.grad_aggr_num)

    def update_policy(self, batch):
        total_opt_result = defaultdict(lambda: 0)

        ppo_epoch = self.policy.custom_config["ppo_epoch"]
        batch_size = batch["value"].shape[
            0
        ]  # XXX(ziyu): suppose the first dimension is batch_size
        num_mini_batch = self.policy.custom_config["num_mini_batch"]  # num_mini_batch
        mini_batch_size = batch_size // num_mini_batch
        assert mini_batch_size > 0

        for _ in range(ppo_epoch):
            rand = torch.randperm(batch_size).numpy()
            for i in range(0, batch_size, mini_batch_size):
                tmp_slice = slice(i, min(batch_size, i + mini_batch_size))
                tmp_batch = get_part_data_from_batch(batch, rand[tmp_slice])

                tmp_opt_result = self.loss(tmp_batch, self.optimize_policy)
                for k, v in tmp_opt_result.items():
                    total_opt_result[k] += v
        # a,b = self.loss(batch, self.optimize_policy)
        self.switch_optimization()
        return total_opt_result

    def update_shaping_function(self, batch):
        total_opt_result = defaultdict(lambda: 0)

        ppo_epoch = self.policy.custom_config["ppo_epoch"]
        batch_size = batch["value"].shape[
            0
        ]  # XXX(ziyu): suppose the first dimension is batch_size
        num_mini_batch = self.policy.custom_config["num_mini_batch"]  # num_mini_batch
        mini_batch_size = batch_size // num_mini_batch
        assert mini_batch_size > 0

        self.h = None
        for _ in range(ppo_epoch):
            rand = torch.randperm(batch_size).numpy()
            for i in range(0, batch_size, mini_batch_size):
                tmp_slice = slice(i, min(batch_size, i + mini_batch_size))
                tmp_batch = get_part_data_from_batch(batch, rand[tmp_slice])

                tmp_opt_result = self.loss(tmp_batch, self.optimize_policy)
                for k, v in tmp_opt_result.items():
                    total_opt_result[k] += v
        # a,b = self.loss(batch, self.optimize_policy)
        self.switch_optimization()

        return total_opt_result

    def preprocess(self, batch, **kwargs):
        """
        build share_obs here
        """

        share_obs = np.concatenate(  # obs shape: batch_size * obs_size
            [batch[pid][Episode.CUR_OBS] for pid in sorted(batch)], axis=1
        )
        for pid, cur_batch in batch.items():
            # if "share_obs" in cur_batch:
            #     continue
            # else:
            cur_batch["share_obs"] = share_obs

        return batch
