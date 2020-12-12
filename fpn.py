#the fpn network for heatmap regression
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import resnet as M
import resnet_fpn as resnet_fpn


# 3*heatmap + 1 origin image(2d tensor) cancatenated along dimension 0

# net = GazeNet()
# net = DataParallel(net)
# net.cuda()
    
# Feature Pyramid Network
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # bottom up
        self.resnet = resnet_fpn.resnet50(pretrained=True)

        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c5_conv = nn.Conv2d(2048, 256, (1, 1))
        self.c4_conv = nn.Conv2d(1024, 256, (1, 1))
        self.c3_conv = nn.Conv2d(512, 256, (1, 1))
        self.c2_conv = nn.Conv2d(256, 256, (1, 1))
        #self.max_pool = nn.MaxPool2d((1, 1), stride=2)

        self.p5_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p4_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p3_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p2_conv = nn.Conv2d(256, 256, (3, 3), padding=1)

        # predict heatmap
        self.sigmoid = nn.Sigmoid()
        self.predict = nn.Conv2d(256, 1, (3, 3), padding=1)
 
    def top_down(self, x):
        c2, c3, c4, c5 = x
        p5 = self.c5_conv(c5)
        p4 = self.upsample(p5) + self.c4_conv(c4)
        p3 = self.upsample(p4) + self.c3_conv(c3)
        p2 = self.upsample(p3) + self.c2_conv(c2)

        p5 = self.relu(self.p5_conv(p5))
        p4 = self.relu(self.p4_conv(p4))
        p3 = self.relu(self.p3_conv(p3))
        p2 = self.relu(self.p2_conv(p2))

        return p2, p3, p4, p5

    def forward(self, x):
        # bottom up
        c2, c3, c4, c5 = self.resnet(x)

        # top down
        p2, p3, p4, p5 = self.top_down((c2, c3, c4, c5))

        heatmap = self.sigmoid(self.predict(p2))
        return heatmap