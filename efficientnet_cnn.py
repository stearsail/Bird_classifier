import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

model = models.efficientnet_b0(weights='DEFAULT')

for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Linear(in_features=num_features,
              out_features=512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(in_features=512, out_features=525)
)

model.to(device = 'cuda')
# summary(model, (3,224,224))