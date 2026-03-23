import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiTaskModel, self).__init__()
        
        # Load ResNet backbone
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove fully connected layer

        # The output size of from self.backbone
        n_features = resnet.fc.in_features

        # TODO: Classification head
        self.classifier = nn.Linear(n_features, num_classes)# YOUR CODE HERE

        # TODO: Localization head (bounding box regression)
        self.regressor = nn.Linear(n_features, 4) # YOUR CODE HERE

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)  # Flatten the output

        # TODO: Model output
        class_out = self.classifier(out) # YOUR CODE HERE for classification output
        bbox_out = self.regressor(out) # YOUR CODE HERE for bounding box regression output

        return class_out, bbox_out
