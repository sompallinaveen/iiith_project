import torch.nn as nn

class CNNEmotion(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Linear(
                32 * 10 * 50,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                7
            )
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.conv(x)

        x = x.view(
            x.size(0),
            -1
        )

        return self.fc(x)