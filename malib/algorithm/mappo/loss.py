# -*- coding: utf-8 -*-
import torch
import gym
from malib.algorithm.common import misc
from malib.algorithm.common.loss_func import LossFunc
from malib.algorithm.mappo.utils import huber_loss, mse_loss, PopArt
from malib.backend.datapool.offline_dataset_server import Episode
from malib.algorithm.common.model import get_model


class MAPPOLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super(MAPPOLoss, self).__init__()

        self.use_popart = False
        self.clip_param = 0.2

        self.value_normalizer = PopArt(1) if self.use_popart else None

        self._use_clipped_value_loss = True
        self._use_huber_loss = True
        if self._use_huber_loss:
            self.huber_delta = 10.0
        self._use_value_active_masks = False
        self._use_policy_active_masks = False

        self._use_max_grad_norm = True
        self.max_grad_norm = 10.0

        self.entropy_coef = 1e-3

        self.use_gae = True

        self.gamma = 0.99
        self.gae_lambda = 0.95

        # ===========params for meta-learning=============================
        self.optim_epochs = 10
        self.lr_phi = 1e-5

    def reset(self, policy, config):
        """Replace critic with a centralized critic"""
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            # self._set_centralized_critic()
            self.setup_optimizers()

    def setup_optimizers(self, *args, **kwargs):
        """Accept training configuration and setup optimizers"""

        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = {
                "actor": optim_cls(
                    self.policy.actor.parameters(), lr=self._params["actor_lr"]
                ),
                "critic": optim_cls(
                    self.policy.critic.parameters(), lr=self._params["critic_lr"]
                ),
                "meta": optim_cls(
                    self.policy.meta.parameters(), lr=self.lr_phi
                ),
            }
        else:
            self.optimizers["actor"].param_groups = []
            self.optimizers["actor"].add_param_group(
                {"params": self.policy.actor.parameters()}
            )
            self.optimizers["critic"].param_groups = []
            self.optimizers["critic"].add_param_group(
                {"params": self.policy.critic.parameters()}
            )
            self.optimizers["meta"].param_groups = []
            self.optimizers["meta"].add_param_group(
                {"params": self.policy.meta.parameters()}
            )

    def __call__(self, sample, optimize_policy=True, update_actor=True):
        if optimize_policy:
            policy_loss, value_loss = self.update_policies(sample)
        else:
            policy_loss, value_loss = self.update_f_function(sample)
        self._policy.opt_cnt += 1


        return dict(
            policy_loss=policy_loss.detach().cpu().numpy(),
            value_loss=value_loss.detach().cpu().numpy(),
        )

    def update_policies(self, sample):
        cast = lambda x: torch.FloatTensor(x.copy()).to(self._policy.device)

        (
            share_obs_batch,
            obs_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            active_masks_batch,
            old_action_probs_batch,
            adv_targ,
            available_actions_batch,
            steps_left,
        ) = (
            cast(sample["share_obs"]),
            cast(sample[Episode.CUR_OBS]),
            cast(sample[Episode.ACTIONS]).long().unsqueeze(-1),
            cast(sample["value"]),
            cast(sample["return"]),
            None,  # cast(sample["active_mask"]),
            cast(sample["action_probs"]),
            cast(sample["advantage"]),
            cast(sample["available_action"]),
            cast(sample["steps_left"]),
        )
        # print("steps_left shape:", steps_left.shape)
        adv_targ = (adv_targ - adv_targ.mean()) / (1e-9 + adv_targ.std())

        values, action_log_probs, dist_entropy = self._evaluate_actions(
            share_obs_batch,
            obs_batch,
            actions_batch,
            available_actions_batch,
            active_masks_batch,
        )
        old_action_log_probs_batch = torch.log(
            old_action_probs_batch.gather(-1, actions_batch)
        )
        imp_weights = torch.exp(
            action_log_probs.unsqueeze(-1) - old_action_log_probs_batch
        )

        surr1 = imp_weights * adv_targ
        surr2 = (
                torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * adv_targ
        )

        if self._use_policy_active_masks:
            policy_action_loss = (
                                         -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                                         * active_masks_batch
                                 ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        self.optimizers["actor"].zero_grad()
        policy_loss = policy_action_loss - dist_entropy * self.entropy_coef
        # policy_loss.backward(retain_graph=True)

        # ================================= for meta-gradient computation ====================================================
        self.optimizers["actor"].zero_grad()
        log_p = action_log_probs
        grad = None
        for j in range(action_log_probs.shape[1]):
            if grad is None:
                grad = torch.autograd.grad(outputs=log_p[:, j].mean(), inputs=self._policy.actor.parameters(), retain_graph=True)
            else:
                grad += torch.autograd.grad(outputs=log_p[:, j].mean(), inputs=self._policy.actor.parameters(), retain_graph=True)
        # ================================= for meta-gradient computation ====================================================
        policy_loss.backward(retain_graph=True)
        if self._use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._policy.actor.parameters(), self.max_grad_norm
            )
        self.optimizers["actor"].step()

        # ============================== Critic optimization ================================
        modified_return_batch = torch.sum(torch.multiply(self._policy.meta(obs_batch), return_batch), dim=-1) # the theta is updated by the shaped return
        # print("modified return shape", modified_return_batch.shape)
        value_loss = self._calc_value_loss(
            values, value_preds_batch, modified_return_batch, active_masks_batch
        )
        self.optimizers["critic"].zero_grad()
        value_loss.backward(retain_graph=True)
        if self._use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._policy.critic.parameters(), self.max_grad_norm
            )
        self.optimizers["critic"].step()

        # =============================== For the h= theta w.r.t. phi computation =============================
        """
                for each state in the state batch
                compute { nabla_{theta} log pi_{theta}(s,a) } and { nabla_{phi} R_{tau}(s,a) }

                1. compute nabla_{phi} R_{tau}(s, a) using the minibatch of the state
                2. compute nabla_{theta} log pi_{theta}(s, a)
                3. conduct matrix multiplication of nabla_{phi} R_{tau}(s, a) and nabla_{theta} log pi_{theta}(s, a)
                4. sum over the multiplication results of all states

                only pick up few samples for computing gradient of theta w.r.t. phi
        """
        self.optimizers["actor"].zero_grad()
        estimated_q = torch.multiply(torch.pow(self.gamma, steps_left), modified_return_batch.unsqueeze(-1))
        # print("estimated_q shape:", estimated_q.shape)
        # log_p = torch.sum(action_log_probs)
        # with torch.autograd.set_detect_anomaly(True):
        # print("estimated_q shape:", estimated_q.shape)
        # log_p = action_log_probs
        # grad = None
        # for j in range(action_log_probs.shape[1]):
        #     if grad is None:
        #         grad = torch.autograd.grad(outputs=log_p[:, j].mean(), inputs=self._policy.actor.parameters())
        #     else:
        #         grad += torch.autograd.grad(outputs=log_p[:, j].mean(), inputs=self._policy.actor.parameters())
        # grad =  torch.autograd.grad(outputs=action_log_probs, inputs=self._policy.actor.parameters(), allow_unused=True)
        for k, weight in enumerate(self._policy.actor.parameters()):
            # if weight.fast is None:
            #     weight.fast = weight - self._params["actor_lr"]  * estimated_q * grad[k]
            # else:
            weight = weight - self._params["actor_lr"]  * estimated_q.mean() * grad[k]

        return policy_loss, value_loss

    def update_f_function(self, sample):
        # compute the gradient of phi and update the meta-network

        cast = lambda x: torch.FloatTensor(x.copy()).to(self._policy.device)
        """
            the loss of shaping weight function f, which is quite complicated

            define optimization of f_phi
            nabla_{phi} J = { nabla_{theta} log pi_{theta}(s,a) } * 
                            { nabla_{phi} theta } * 
                            Q_{True}(s, a)

                            = { nabla_{theta} log pi_{theta}(s,a) } * 
                            alpha * { nabla_{theta'} log pi_{theta'}(s',a') } * { nabla_{phi} R_{tau} } * 
                            Q_{True}(s, a)

                            = { nabla_{theta} log pi_{theta}(s,a) } * 
                            alpha * { nabla_{theta'} log pi_{theta'}(s',a') } * 
                            { sum_{i=0}^{|tau|-1 } gamma^i F(s_i,a_i) nabla_{phi} f_{phi}(s_i) } *
                            Q_{True}(s, a)
        """
        (
            share_obs_batch,
            obs_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            active_masks_batch,
            old_action_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = (
            cast(sample["share_obs"]),
            cast(sample[Episode.CUR_OBS]),
            cast(sample[Episode.ACTIONS]).long().unsqueeze(-1),
            cast(sample["value"]),
            cast(sample["return"]),
            None,  # cast(sample["active_mask"]),
            cast(sample["action_probs"]),
            cast(sample["advantage"]),
            cast(sample["available_action"]),
        )


        """
            Here we do a bunch of optimization epochs over the data
        """
        for _ in range(self.optim_epochs):

            adv_targ = (adv_targ - adv_targ.mean()) / (1e-9 + adv_targ.std())

            values, action_log_probs, dist_entropy = self._evaluate_actions(
                share_obs_batch,
                obs_batch,
                actions_batch,
                available_actions_batch,
                active_masks_batch,
            )
            old_action_log_probs_batch = torch.log(
                old_action_probs_batch.gather(-1, actions_batch)
            )
            imp_weights = torch.exp(
                action_log_probs.unsqueeze(-1) - old_action_log_probs_batch
            )

            surr1 = imp_weights * adv_targ
            surr2 = (
                    torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
            )

            if self._use_policy_active_masks:
                policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                                             * active_masks_batch).sum() / active_masks_batch.sum()
            else:
                policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

            policy_loss = policy_action_loss - dist_entropy * self.entropy_coef

            if self._use_max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self._policy.actor.parameters(), self.max_grad_norm
                )

            true_pol_loss = policy_loss

            self.optimizers["meta"].zero_grad()
            true_pol_loss.backward()
            self.optimizers["meta"].step()

            return_batch = torch.mean(return_batch, dim=-1)
            value_loss = self._calc_value_loss(
                values, value_preds_batch, return_batch, active_masks_batch
            )

        return true_pol_loss, value_loss

    def _evaluate_actions(
        self,
        share_obs_batch,
        obs_batch,
        actions_batch,
        available_actions_batch,
        active_masks_batch=None,
    ):
        assert active_masks_batch is None, "Not handle such case"

        logits = self._policy.actor(obs_batch)
        logits -= 1e10 * (1 - available_actions_batch)

        dist = torch.distributions.Categorical(logits=logits)
        # TODO(ziyu): check the shape!!!
        action_log_probs = dist.log_prob(
            actions_batch.view(logits.shape[:-1])
        )  # squeeze the last 1 dimension which is just 1
        dist_entropy = dist.entropy().mean()

        values = self._policy.critic(share_obs_batch)

        return values, action_log_probs, dist_entropy

    def _calc_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch=None
    ):
        if self.use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            # print("value return batch shape", return_batch.shape)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss


    def zero_grad(self):
        pass

    def step(self):
        pass