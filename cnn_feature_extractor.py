import torch.nn as nn
import torch.nn.functional as F
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class BaseFeaturesExtractor(nn.Module):
    def __init__(self, observation_space: gym.Space, features_dim, **kwargs):
        super().__init__(**kwargs)
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim


class SonicCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, n_outputs: int = None, **kwargs):
        super().__init__(observation_space, features_dim, **kwargs)
        self.n_outputs = n_outputs
        input_shape = observation_space.shape # assuming (1, 112, 160)
        conv_out_spat_size = input_shape[1]//(4*2) * input_shape[2]//(4*2) # assuming 20 * 14
        # must be same padding to get expect dim of linear
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=9, stride=4, padding=(4,4))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=(2,2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same")
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64*conv_out_spat_size, features_dim)
        if n_outputs:
            self.head = nn.Linear(features_dim, n_outputs)

    def forward(self, observations):
        x = observations / 128
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        if self.n_outputs:
            x = F.relu(x)
            x = self.head(x)
        return x
        
