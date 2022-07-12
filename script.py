import torch
from torchvision import models
model = models.resnet18(pretrained = True)
torch.save(model.state_dict(), "torchserve-demo/resnet18.pth")
torchscripted_model = torch.jit.script(model)
torchscripted_model.save('resnet18.pt')
