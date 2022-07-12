import torch
from torchvision import models
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
torch.save(model.state_dict(), "resnet18.pth")
torchscripted_model = torch.jit.script(model)
torchscripted_model.save('resnet18.pt')
