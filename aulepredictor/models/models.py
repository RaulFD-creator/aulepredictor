import torch.nn as nn
import eztorch4conv as ez
from .layers import *

class Aule_General(nn.Module):
    def __init__(self):
        super().__init__()
        ACTIVATION_FUNCTION = nn.ReLU()
        self.features = nn.Sequential(
            conv3d(in_channels=6, out_channels=32, kernel_size=5, stride=1, dropout=0,
                                batch_norm=False, padding='same', activation_function=ACTIVATION_FUNCTION,
                                ),
            conv3d(in_channels=32, out_channels=48, kernel_size=4, stride=1, dropout=0, 
                                batch_norm=True, padding='valid', activation_function=ACTIVATION_FUNCTION),
            nn.MaxPool3d(kernel_size=2),

            conv3d(in_channels=48, out_channels=64, kernel_size=4, stride=1, dropout=0, 
                                batch_norm=True, padding='same', activation_function=ACTIVATION_FUNCTION),
            conv3d(in_channels=64, out_channels=96, kernel_size=4, stride=1, dropout=0, 
                                batch_norm=True, padding='valid', activation_function=ACTIVATION_FUNCTION),
            nn.MaxPool3d(kernel_size=2))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            dense(in_features=768, out_features=256, dropout=0,
                                activation_function=ACTIVATION_FUNCTION, batch_norm=True),
            dense(in_features=256, out_features=1, dropout=0,
                                activation_function=nn.Sigmoid(), batch_norm=False)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)
