import copy
import time

import cv2
import numpy
import ray
import torch

import models


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        print('Ray gpus', ray.get_gpu_ids())
        print('Torch gpu:', torch.cuda.is_available())
        import os
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        # torch.cuda.device(ray.get_gpu_ids()[0])
        # print("Current device", torch.cuda.current_device())
        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.cpu()
        # self.model.cuda(0)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        print("Model device", str(next(self.model.parameters()).device))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        # self.model.cuda(ray.get_gpu_ids()[0])
        self.model.train()
        print("Model device", str(next(self.model.parameters()).device))

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.add_weight_decay(self.model, self.config.weight_decay),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                nesterov=True,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        min_games = 1
        while ray.get(shared_storage.get_info.remote("num_played_games")) < min_games:
            time.sleep(0.1)

        pipelined_batch = replay_buffer.get_batch.remote()

        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            # print('Start train loop')
            index_batch, batch = ray.get(pipelined_batch)
            # print('Got batch')
            pipelined_batch = replay_buffer.get_batch.remote()
            # print(f'Index {index_batch}')
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
                grad_norm,
                l2_norm,
            ) = self.update_weights(batch)
            # print('Weights updated')

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                    "grad_norm": grad_norm,
                    "l2_norm": l2_norm,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio_max:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio_max
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def update_weights(self, batch):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch

        # print('Update weights')
        # print(f'Weight {weight_batch[:8]}')
        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        # observation_numpy = numpy.array(observation_batch, dtype=numpy.float32)
        # observation_batch = torch.tensor(observation_numpy).float().to(device)
        observation_batch = torch.from_numpy(observation_batch.copy()).to(device)
        # observation_batch = torch.tensor(observation_batch).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        # observation_batch: batch, num_unroll_steps+1, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        channels, height, width = self.config.observation_shape

        print(f'Ori {target_policy[:4].detach().cpu().numpy()}')
        if self.training_step < 100:
            # target_value *= 0.0
            target_policy = torch.full_like(target_policy, 1 / len(self.config.action_space)).float().to(device)


        target_value = models.scalar_to_support(target_value, self.config.support_size)
        mau = 0
        # for i in range(target_reward.shape[0]):
        #     if target_reward[i][1:].sum() > 0.5:
        #         mau = i
        target_reward = models.scalar_to_support(
            target_reward, self.config.support_size
        )
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        ## Generate predictions
        value, reward, policy_logits, hidden_state = self.model.initial_inference(
            observation_batch[:, 0]
        )
        predictions = [(value, reward, policy_logits)]
        batch_size = observation_batch.shape[0]
        ori_hidden_state = hidden_state.detach()
        next_loss = 0
        debug_hist = [(
            models.support_to_scalar(torch.log(target_value[0:1, 0]), self.config.support_size).item(),
            models.support_to_scalar(value[0:1], self.config.support_size).item(),
        )]

        prediction_img = numpy.zeros((3, 96 * 2, 96 * action_batch.shape[1]), dtype=numpy.uint8)

        def append_img(img, r, c):
            prediction_img[:, 96*r:96*(r+1), 96*c:96*(c+1)] = numpy.clip(img.detach().cpu().numpy() * 255, 0, 255)

        append_img(observation_batch[0, 0, :channels], 0, 0)


        vae_state = self.model.vae.represent(observation_batch[:, 0])
        pred = self.model.vae.decode(vae_state)
        append_img(pred[0], 1, 0)
        next_loss = torch.nn.MSELoss(reduction='none')(observation_batch[:, 0, :channels], pred).view(batch_size, -1).mean(dim=1)
        print(0, next_loss.mean().item())
        print(self.model.vae.decoder.last_mean, self.model.vae.decoder.last_var)

        for i in range(1, action_batch.shape[1]):
            # print(f'>>>unroll {i}')
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                hidden_state, action_batch[:, i]
            )
            if mau is not None:
                tv = models.support_to_scalar(torch.log(target_value[mau:mau+1, i]), self.config.support_size).item()
                rv = models.support_to_scalar(value[mau:mau+1], self.config.support_size).item()
                tr = models.support_to_scalar(torch.log(target_reward[mau:mau+1, i]), self.config.support_size).item()
                rr = models.support_to_scalar(reward[mau:mau+1], self.config.support_size).item()
                tp = target_policy[mau, i]
                debug_hist.append((tv, rv, tr, rr))
                # print(f'Weight {weight_batch[mau].item()}')
                # print(f'Target value {tv:.6f}')
                # print(f'Value {rv:.6f}')
                # print(f'Target reward {tr:.6f}')
                # print(f'Reward {rr:.6f}')
                # print(f'Target policy {tp}')
                # print('!At 0')
                # print(f'!Target value {models.support_to_scalar(torch.log(target_value[0:1, i]), self.config.support_size).item()}')
                # print(f'!Value {models.support_to_scalar(value[0:1], self.config.support_size).item()}')
                # print(f'!Reward {models.support_to_scalar(reward[0:1], self.config.support_size).item()}')

            # pred = self.model.vae_test(observation_batch[:, i])#, :channels])

            # vae_state = self.model.vae.recurrent(vae_state, action_batch[:, i], observation_batch[:, i, :channels])
            vae_state = self.model.vae.recurrent(vae_state, action_batch[:, i], observation_batch[:, i, :channels])
            pred = self.model.vae.decode(vae_state)
            current_loss = torch.nn.MSELoss(reduction='none')(observation_batch[:, i, :channels], pred).view(batch_size, -1)
            # current_loss = torch.clamp(current_loss, min=0.0002)
            current_loss = current_loss.mean(dim=1)
            next_loss += current_loss
            print(i, current_loss.mean().item())
            print(self.model.vae.decoder.last_mean, self.model.vae.decoder.last_var)
            print('Last he', self.model.vae.last_he)
            append_img(observation_batch[0, i, :channels], 0, i)
            append_img(pred[0], 1, i)

            # with torch.no_grad():
            #     target_state = self.model.representation(observation_batch[:, i])
            # target_state.detach_()

            # target_state = self.model.representation(observation_batch[:, i])

            # with torch.no_grad():
            #     current_value, _, current_policy, _ = self.model.initial_inference(observation_batch[:, i])

            # value_mix = 0.0
            # policy_mix = 0.0 if self.training_step < 100 else 0.0
            # print('Replace',
            #         models.support_to_scalar(torch.log(target_value[0:1, i]), self.config.support_size).item(),
            #         'with',
            #         models.support_to_scalar(torch.log(torch.softmax(current_value[0:1], dim=1)), self.config.support_size).item(),
            #         'mix',
            #         models.support_to_scalar(torch.log(
            #             target_value[0:1, i] * (1. - value_mix) + torch.softmax(current_value[0:1], dim=1) * value_mix
            #         ), self.config.support_size).item(),
            # )
            # print('Replace policy',
            #         target_policy[0:1, i].detach().cpu().numpy(),
            #         'with',
            #         torch.softmax(current_policy[0:1], dim=1).detach().cpu().numpy(),
            # )
            # target_value[:, i] = target_value[:, i] * (1. - value_mix) + torch.softmax(current_value, dim=1) * value_mix
            # target_policy[:, i] = target_policy[:, i] * (1. - policy_mix) + torch.softmax(current_policy, dim=1) * policy_mix

            # current_loss = torch.nn.MSELoss(reduction='none')(hidden_state, target_state).view(batch_size, -1).mean(dim=1)
            # if i == 1:
            #     next_loss = current_loss
            # elif i < observation_batch.shape[1]:
            #     next_loss += current_loss
            # else:
            #     print(f"Warning OOB {i} {observation_batch.shape[1]}")
            #     assert False
            # print(i, current_loss.mean().item(), torch.nn.MSELoss()(hidden_state, ori_hidden_state).item(),
            #         torch.nn.MSELoss()(target_state, ori_hidden_state).item())
            # print(f"mn {current_loss.view(len(current_loss), -1).mean(1).min()} mx {current_loss.view(len(current_loss), -1).mean(1).max()}")
            # print(f"0-1: {torch.nn.MSELoss()(hidden_state[0], hidden_state[1]).item()}")
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))#, next_value, next_policy_logits))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        if mau is not None:
            print(f'Debug hist {debug_hist}')
            print(f'Policy {torch.softmax(predictions[0][2][0:1], dim=1).detach().cpu().numpy()}')
            print(f'Target {target_policy[0:1, 0].detach().cpu().numpy()}')

        cv2.imwrite('predicted.jpg', cv2.cvtColor(
            numpy.moveaxis(numpy.clip(prediction_img, 0, 255), 0, -1), cv2.COLOR_RGB2BGR,
        ))

        ## Compute losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)
        value, reward, policy_logits = predictions[0]
        # Ignore reward loss for the first batch step
        current_value_loss, _, current_policy_loss = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        # Compute priorities for the prioritized replay (See paper appendix Training)
        pred_value_scalar = (
            models.support_to_scalar(value, self.config.support_size)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
        priorities[:, 0] = (
            numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
            ** self.config.PER_alpha
        )

        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
            ) = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
            )
            # if mau is not None:
            #     print('reward_loss', current_reward_loss[mau].mean().item())

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            current_value_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_reward_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                models.support_to_scalar(value, self.config.support_size)
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            priorities[:, i] = (
                numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + 2.0 * reward_loss + policy_loss# + 10.0 * next_loss
        # loss = reward_loss + value_loss * self.config.value_loss_weight #+ 0.01 * next_loss
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch

        loss += 100.0 * next_loss
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        vae_params = []
        grad_norm = 0.
        for n, p in self.model.named_parameters():
            if p.grad is None:
                print('No grad!!!!')
                print(n)
            if 'vae' in n:
                grad_norm += p.grad.data.norm(2) ** 2
                vae_params.append(p)

        print('VAE gradnorm', (grad_norm ** (1. / 2)).item())

        torch.nn.utils.clip_grad_norm_(vae_params, 2)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 8)

        grad_norm = 0.
        l2_norm = 0.
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
            l2_norm += torch.norm(p).item()

        grad_norm = grad_norm ** (1. / 2)
        l2_norm *= self.config.weight_decay

        # for name, param in self.model.named_parameters():
        #     print(name, param.grad.abs().mean().item(), param.data.abs().mean().item())
 
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
            grad_norm,
            l2_norm,
        )

    def update_lr(self):
        """
        Update learning rate
        """
        warmup = 100
        if self.training_step < warmup:
            lr = self.config.lr_init * (self.training_step + 1) / warmup
        else:
            lr = self.config.lr_init * self.config.lr_decay_rate ** (
                self.training_step / self.config.lr_decay_steps
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        return value_loss, reward_loss, policy_loss

    @staticmethod
    def add_weight_decay(net, l2_value, skip_list=()):
        decay, no_decay = [], []
        for name, param in net.named_parameters():
            if not param.requires_grad:
                print(f'FROZEN {name}') # frozen weights
                continue
            if (len(param.shape) == 1 or name.endswith(".bias") or "bn" in name or name in skip_list):# and ("fc" not in name):
                no_decay.append(param)
                print('-', name)
            else:
                decay.append(param)
                print('+', name)
        return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}] 
