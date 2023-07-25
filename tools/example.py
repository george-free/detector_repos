import torch.optim as optim
import torch.nn as nn

# Define a neural network
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(32 * 6 * 6, 10)
        
        self.mm1 = nn.ModuleList(
            self.conv1,
            self.bn1,
            self.relu1,
        )
        self.mm2 == nn.ModuleList(
            self.conv2,
            self.bn2,
            self.relu2
        )

    def forward(self, x):
        for layer in self.mm1:
            x = layer(x)
        for layer in self.mm2:
            x = layer(x)
        x = x.view(-1, 32 * 6 * 6)
        x = self.fc(x)
        return x

# Create an instance of the neural network
net = MyNet()

# Define the optimizer
params_with_decay = []
params_without_decay = []
for name, param in net.named_parameters():
    if 'bias' in name:
        params_without_decay.append(param)
    else:
        params_with_decay.append(param)

optimizer = optim.SGD([
    {'params': params_with_decay, 'weight_decay': 0.0005},
    {'params': params_without_decay, 'weight_decay': 0},
    {'params': net.fc.parameters(), 'lr': 0.01}
], lr=0.001, momentum=0.9)

# Train the neural network
# ...
