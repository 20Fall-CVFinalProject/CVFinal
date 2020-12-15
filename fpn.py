#the fpn network for heatmap regression
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch


# 3*heatmap + 1 origin image(2d tensor) cancatenated along dimension 0

# net = GazeNet()
# net = DataParallel(net)
# net.cuda()
    
# Feature Pyramid Network

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # map pool
        #self.map_pool = nn.AvgPool2d(4, stride=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # add gaze_field mat
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        p2 = self.layer1(x)
        p3 = self.layer2(p2)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)

        return p2, p3, p4, p5

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    # return model
    return model

class BottleNeck(nn.Module):
    expansion = 4  

    def __init__(self, indepth, depth, stride=1, downsample=None, size=64):
        super(BottleNeck, self).__init__() # why
        self.conv1 = conv1x1(indepth, depth)
        self.conv2 = conv3x3(depth, depth, stride)
        self.conv3 = conv1x1(depth, depth*4)
        self.bn = nn.BatchNorm2d(depth)
        self.bn_4 = nn.BatchNorm2d(depth * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        input = x

        y = self.conv1(x)
        y = self.bn(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn(y)
        y = self.relu(y)
        
        y = self.conv3(y)
        y = self.bn_4(y)

        if self.downsample is not None:
            input = self.downsample(input)
        
        return self.relu(y+input)


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # bottom up
        self.resnet = resnet50(pretrained=True)

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

        re5 = self.relu(self.p5_conv(p5))
        re4 = self.relu(self.p4_conv(p4))
        re3 = self.relu(self.p3_conv(p3))
        re2 = self.relu(self.p2_conv(p2))

        return re2, re3, re4, re5

    def forward(self, x):
        c2, c3, c4, c5 = self.resnet(x)

        r2, r3, r4, r5 = self.top_down((c2, c3, c4, c5))

        activation = self.sigmoid(self.predict(r2))


        return activation


if __name__ == '__main__':
	net = resnet50()
	print(net)