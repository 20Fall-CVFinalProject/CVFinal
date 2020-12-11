import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()

def resnet50(pretrained=False, **kwargs):
    model = ResNet()
    return model
