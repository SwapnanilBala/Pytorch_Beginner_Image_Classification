import torch.nn as nn


class SmallCNN(nn.Module):
    """
    A tiny CNN that’s easy to understand and trains fast on CIFAR-10.
    Input:  (N, 3, 32, 32)
    Output: (N, 10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()


# Here we are extracting the visual features from images using convolutions + ReLU + pooling:
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # -> (32, 32, 32) -> here we take an RGB image and produce 32 feature maps
            nn.ReLU(inplace=True), # This adds non-linearity so the network can learn complex patterns.
            nn.MaxPool2d(2),  # downsampling by 2 -> reducing 32 X 32 to 16 X 16                           # -> (32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (64, 16, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # same here -> from 16 X 16 to 8 X 8,   # -> (64, 8, 8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# -> (128, 8, 8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # -> (128, 4, 4)
        )
        # This classifier here converts those features into class scores
        self.classifier = nn.Sequential(
            nn.Flatten(),  # vector transformation    # -> (128*4*4)
            nn.Linear(128 * 4 * 4, 256), # learns a dense representation
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    # This defines how the input flows: input -> features -> classifiers -> logits
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
