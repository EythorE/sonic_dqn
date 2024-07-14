import torch.nn as nn

class SonicCNN(nn.Module):
    def __init__(self, observation_space, features_dim=512, n_outputs: int = None):
        super().__init__()
        self.n_outputs = n_outputs
        input_shape = observation_space.shape # assuming (1, 112, 160)
        conv_out_spat_size = input_shape[1]//(4*2) * input_shape[2]//(4*2) # assuming 20 * 14
        # must be same padding to get expect dim of linear
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=9, stride=4, padding=(4,4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=(2,2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same")
        self.fc = nn.Linear(64*conv_out_spat_size, features_dim)
        if n_outputs:
            self.head = nn.Linear(features_dim, n_outputs)
        self.relu = nn.functional.relu

    def forward(self, observations):
        x = observations / 128
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        if self.n_outputs:
            x = self.relu(x)
            x = self.head(x)
        return x
        
