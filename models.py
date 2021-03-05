import math
from abc import ABC, abstractmethod

import torch


class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
            )
        elif config.network == "jago":
            return MuZeroJagoNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.jago_blocks,
                config.jago_encoding_size,
                config.jago_fc_reward_layers,
                config.jago_fc_value_layers,
                config.jago_fc_policy_layers,
                config.support_size,
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        return encoded_state
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        return next_encoded_state, reward

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias
    )

def conv2d_init(m):
    assert isinstance(m, torch.nn.Conv2d)
    # if m.kernel_size[0] == 1:
    #     return
    # assert m.kernel_size[0] == 3 and m.kernel_size[1] == 3
    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))

def normalize_state(state):
    min_state = torch.flatten(state, start_dim=1).min(1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    max_state = torch.flatten(state, start_dim=1).max(1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    scale = max_state - min_state
    normalized_state = (state - min_state) / scale
    return normalized_state

def normalize_state_fc(state):
    assert len(state.shape) == 2
    min_state = state.min(1, keepdim=True)[0]
    max_state = state.max(1, keepdim=True)[0]
    scale = max_state - min_state
    normalized_state = (state - min_state) / scale
    return normalized_state


G = 8

# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1, identity_init=False):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride, bias=False)
        self.bn1 = torch.nn.GroupNorm(G, num_channels)
        self.conv2 = conv3x3(num_channels, num_channels, bias=False)
        self.bn2 = torch.nn.GroupNorm(G, num_channels)
        if identity_init:
            self.bn2.weight.data.fill_(0)

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += inp
        x = torch.nn.functional.relu(x)
        return x

    def forward2(self, x):
        inp = x
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x += inp
        return x

class ResidualBlock2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, identity_init=False):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, bias=False)
        self.bn1 = torch.nn.GroupNorm(G, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, bias=False)
        self.bn2 = torch.nn.GroupNorm(G, out_channels)
        if identity_init:
            self.bn2.weight.data.fill_(0)

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += inp[:, :x.shape[1]]
        x = torch.nn.functional.relu(x)
        return x

    def forward2(self, x):
        inp = x
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x += inp[:, :x.shape[1]]
        return x


# Convolution block
class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, zero_init=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = torch.nn.GroupNorm(G if out_channels >= G * 2 else out_channels // 2, out_channels)
        if zero_init:
            self.bn.weight.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        return x

# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvolutionBlock(
            in_channels,
            out_channels // 2,
            stride=2,
        )
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)]
        )
        self.conv2 = ConvolutionBlock(
            out_channels // 2,
            out_channels,
            stride=2,
        )
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_w):
        super().__init__()
        mid_channels = out_channels // 2
        # 96 -> 24 (k=8, p=2, s=4)
        # 24 -> 11 (k=4, p=0, s=2)
        # 11 -> 6 (k=3, p=1, s=2)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=8, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')

        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        # self.bn_state = torch.nn.GroupNorm(G, num_channels)

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        # x = self.bn_state(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        action_space_size,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.conv = ConvolutionBlock(
            num_channels + action_space_size,
            num_channels,
        )
        # self.conv1 = conv3x3(num_channels + action_space_size, num_channels, bias=False)
        # self.bn1 = torch.nn.GroupNorm(G, num_channels)
        # self.conv2 = conv3x3(num_channels, num_channels, bias=False)
        # self.bn2 = torch.nn.GroupNorm(G, num_channels)
        # self.bn2.weight.data.fill_(0)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels, identity_init=False) for _ in range(num_blocks)]
        )
        # self.hidden_state_conv = ConvolutionBlock(num_channels, num_channels, relu=False, zero_init=True)

        # self.bn_state = torch.nn.GroupNorm(G, num_channels)
        # self.bn_reward = torch.nn.GroupNorm(G, num_channels)
        self.conv1x1_reward = torch.nn.Conv2d(num_channels, reduced_channels_reward, 1)
        # self.avg_reward = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv1x1_reward = ConvolutionBlock(num_channels, reduced_channels_reward, kernel_size=1, padding=0)
        self.block_output_size_reward = block_output_size_reward
        # self.fc = mlp(reduced_channels_reward, fc_reward_layers, full_support_size)
        self.fc = mlp(
            self.block_output_size_reward, fc_reward_layers, full_support_size, zero_last=True
        )

    def forward(self, x):
        # act = x[:, self.num_channels:]
        # x = x[:, :self.num_channels]
        # inp = x
        x = self.conv(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = torch.nn.functional.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x += inp[:, :x.shape[1]]
        # x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            # x = block(torch.cat((x, act), 1))
            x = block(x)
        # x = self.bn_state(x)
        # x = self.hidden_state_conv(x)
        # x += inp[:,:x.shape[1]]
        # x = x[:, :self.num_channels]
        state = x
        # x = self.bn_reward(x)
        x = self.conv1x1_reward(x)
        x = torch.nn.functional.relu(x)
        # x = self.avg_reward(x)
        # x = torch.flatten(x, start_dim=1)
        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        super().__init__()
        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        # self.avg_value = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv1x1_value = ConvolutionBlock(num_channels, reduced_channels_value, kernel_size=1, padding=0)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        # self.avg_policy = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv1x1_policy = ConvolutionBlock(num_channels, reduced_channels_policy, kernel_size=1, padding=0)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        # self.fc_value = mlp(reduced_channels_value, fc_value_layers, full_support_size)
        # self.fc_policy = mlp(reduced_channels_policy, fc_policy_layers, action_space_size)
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size, zero_last=True
        )
        self.fc_policy = mlp(
            self.block_output_size_policy, fc_policy_layers, action_space_size,
        )

    def forward(self, x):
        value = self.conv1x1_value(x)
        value = torch.nn.functional.relu(value)
        # value = self.avg_value(value)
        policy = self.conv1x1_policy(x)
        policy = torch.nn.functional.relu(policy)
        # policy = self.avg_policy(policy)
        # value = torch.flatten(value, start_dim=1)
        # policy = torch.flatten(policy, start_dim=1)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class Encoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        depth,
    ):
        super().__init__()
        # 96 -> 23 (k=8, s=4, p=0)
        # 23 -> 10 (k=4, s=2, p=0)
        # 10 -> 4 (k=4, s=2, p=0)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, depth, kernel_size=8, stride=4, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(depth, depth * 2, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(
        self,
        state_size,
        depth,
        out_channels,
    ):
        super().__init__()
        # 1 -> 3 (k=3, s=2)
        # 3 -> 9 (k=5, s=2)
        # 9 -> 21 (k=5, s=2)
        # 21 -> 46 (k=6, s=2)
        # 46 -> 96 (k=6, s=2)
        self.depth = depth
        self.fc = torch.nn.Linear(state_size, depth * 32)
        self.features = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(depth * 32, depth * 8, kernel_size=3, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(depth * 8, depth * 4, kernel_size=5, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(depth * 4, depth * 2, kernel_size=5, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(depth * 2, depth, kernel_size=6, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(depth, out_channels, kernel_size=6, stride=2),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.depth * 32, 1, 1)
        x = self.features(x)
        return x


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        support_size,
        downsample,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        block_output_size_reward = (
            (
                reduced_channels_reward
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        self.representation_network = RepresentationNetwork(
            observation_shape,
            stacked_observations,
            num_blocks,
            num_channels,
            downsample,
        )

        self.dynamics_network = DynamicsNetwork(
            num_blocks,
            num_channels,
            self.action_space_size,
            reduced_channels_reward,
            fc_reward_layers,
            self.full_support_size,
            block_output_size_reward,
        )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            self.full_support_size,
            block_output_size_value,
            block_output_size_policy,
        )

        # VAE
        self.hidden = 600
        self.depth = 64
        self.deter_size = 0
        self.stoch_size = 600
        self.encoder = Encoder(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            self.depth,
        )

        self.fc_post = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * self.depth * 4, self.hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden, self.stoch_size),
            torch.nn.ReLU(inplace=True),
        )

        self.decoder = Decoder(
            self.deter_size + self.stoch_size,
            self.depth,
            observation_shape[0],
        )

        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         conv2d_init(m)

    def vae_test(self, observation):
        embed = self.encoder(observation)
        post = self.fc_post(embed)
        pred = self.decoder(post)
        return pred

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        encoded_state = normalize_state(encoded_state)
        return encoded_state

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        action_one_hot.unsqueeze_(-1)
        action_one_hot.unsqueeze_(-1)
        action_one_hot = action_one_hot.expand(-1, -1, encoded_state.shape[2], encoded_state.shape[3])

        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(x)
        # next_encoded_state = self.hidden_state_bn(next_encoded_state)
        next_encoded_state = normalize_state(next_encoded_state)
        return next_encoded_state, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


########### End ResNet ###########
##################################


class FCBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        assert out_channels >= G * 2
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels, bias=False)
        # self.bn = torch.nn.GroupNorm(G, out_channels)

    def forward(self, x):
        x = self.fc(x)
        # x = self.bn(x)
        x = torch.nn.functional.relu(x)
        return x


class FCResidualBlock(torch.nn.Module):
    def __init__(self, num_channels):
        assert num_channels >= G * 2
        super().__init__()
        self.fc1 = torch.nn.Linear(num_channels, num_channels)
        self.bn1 = torch.nn.GroupNorm(G, num_channels)
        self.fc2 = torch.nn.Linear(num_channels, num_channels)
        self.bn2 = torch.nn.GroupNorm(G, num_channels)

    def forward(self, x):
        inp = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x += inp
        x = torch.nn.functional.relu(x)
        return x


class FCOutput(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, mid_channels)
        # self.bn = torch.nn.GroupNorm(G, mid_channels)
        self.fc2 = torch.nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


class RepresentationJago(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        encoding_size,
    ):
        super().__init__()
        num_channels = 256
        num_blocks = 8
        self.conv1 = ConvolutionBlock(
            observation_shape[0] * (stacked_observations + 1)
            + stacked_observations,
            num_channels // 2,
            stride=2,
        )
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(num_channels // 2) for _ in range(2)]
        )
        self.conv2 = ConvolutionBlock(
            num_channels // 2,
            num_channels,
            stride=2,
        )
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(3)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(3)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks4 = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        self.conv3 = ConvolutionBlock(
            num_channels,
            num_channels * 2,
            stride=2,
        )
        # self.fc = mlp(3 * 3 * num_channels * 2, [], encoding_size, output_activation=torch.nn.ReLU)
        self.fc = FCBlock(3 * 3 * num_channels * 2, encoding_size)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        for block in self.resblocks4:
            x = block(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class RepresentationJagoCnn(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        fc_representation_layers,
        encoding_size,
    ):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels // 2, kernel_size=8, stride=4, padding=0), # 96 -> 23
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels // 2, mid_channels, kernel_size=5, stride=2, padding=0), # 23 -> 10
            torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=0), # 10 -> 8
            torch.nn.ReLU(inplace=True),
        )
        self.conv_output_size = 8 * 8 * mid_channels
        self.fc = mlp(self.conv_output_size, fc_representation_layers, encoding_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.conv_output_size)
        x = self.fc(x)
        return x


class DynamicsJago(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        encoding_size,
        action_space_size,
    ):
        super().__init__()
        self.fc = FCBlock(encoding_size + action_space_size, encoding_size)
        self.resblocks = torch.nn.ModuleList(
            [FCResidualBlock(encoding_size) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.fc(x)
        for block in self.resblocks:
            x = block(x)
        return x


class MuZeroJagoNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = RepresentationJago(
            observation_shape,
            stacked_observations,
            encoding_size,
        )

        # self.representation_network = torch.nn.DataParallel(
        #     RepresentationJagoCnn(
        #         observation_shape[0] * (stacked_observations + 1)
        #         + stacked_observations,
        #         256,
        #         fc_representation_layers,
        #         encoding_size,
        #     )
        # )

        self.dynamics_network = DynamicsJago(
            num_blocks,
            encoding_size,
            action_space_size,
        )

        self.dynamics_reward_network = FCOutput(
            encoding_size, fc_reward_layers, self.full_support_size
        )
        self.prediction_policy_network = FCOutput(
            encoding_size, fc_policy_layers, self.action_space_size
        )
        self.prediction_value_network = FCOutput(
            encoding_size, fc_value_layers, self.full_support_size
        )

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        encoded_state = normalize_state_fc(encoded_state)
        return encoded_state

    def dynamics(self, encoded_state, action):
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_network(x)
        next_encoded_state = normalize_state_fc(next_encoded_state)

        reward = self.dynamics_reward_network(next_encoded_state)

        return next_encoded_state, reward

    def initial_inference(self, observation):
        encoded_state = self.representation_network(observation)
        policy_logits, value = self.prediction(encoded_state)
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
    zero_last=False,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    res = torch.nn.Sequential(*layers)
    if zero_last:
        res[-2].weight.data.fill_(0)
        res[-2].bias.data.fill_(0)
        # res[-2].bias.data[output_size // 2] = 1
    return res


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
