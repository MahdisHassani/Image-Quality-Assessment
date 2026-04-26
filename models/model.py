import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = True
        
    model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

    return model