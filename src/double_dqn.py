"""
Double DQN (van Hasselt et al., 2016) on top of Stable-Baselines3 DQN.

SB3's DQN is vanilla DQN: the TD target takes max over the TARGET network,
    y = r + gamma * max_a Q_target(s', a)
which is known to over-estimate action values. Double DQN decouples action
selection (ONLINE network) from action evaluation (TARGET network):
    a* = argmax_a Q_online(s', a)
    y  = r + gamma * Q_target(s', a*)

Everything else (replay buffer, epsilon-greedy exploration, Huber loss,
target network updates) is identical to SB3 DQN, so checkpoints remain
loadable with either class (identical parameter structure).
"""

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import DQN


class DoubleDQN(DQN):
    """DQN with the Double-DQN TD target. Drop-in replacement for sb3 DQN."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # DOUBLE DQN target:
                # 1) select next actions with the ONLINE network
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                # 2) evaluate them with the TARGET network
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations), dim=1, index=next_actions
                )
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
