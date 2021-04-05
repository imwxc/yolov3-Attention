from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer_noSeq(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_conv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layer1= nn.Linear(channel,channel // reduction,bias=False)
        self.layer2=nn.ReLU(inplace=True)
        self.layer3=nn.Linear(channel // reduction, channel,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.layer1(y)
        y=self.layer2(y)
        y=self.layer3(y)
        y=self.sigmoid(y)
        y=y.view(b, c, 1, 1)
        return x * y.expand_as(x)