import torch.nn as nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False),    # 28 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 26
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),   # 26 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 24
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False),   # 24 + 2*0 - 1*(1 - 1) - 1 / 1 + 1 = 24
            nn.MaxPool2d(kernel_size=2, stride=2)                                   # 24 / 2 = 12
        )   

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),   # 12 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 10
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False),  # 10 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 8
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False),  # 8 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 6
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.avg_pool(x)
        x = self.conv_block_3(x)
        x = x.view(-1, 10)  # Flatten the tensor
        return F.log_softmax(x, dim=-1)
