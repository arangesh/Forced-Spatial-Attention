import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))], 1)


class Attention(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Attention, self).__init__()
        self.weight = 1.0
        self.inplanes = inplanes
        self.attention = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.activation = nn.Sigmoid()
        self.out = nn.Conv2d(2*inplanes, outplanes, kernel_size=1)
        self.out_activation = nn.ReLU(inplace=True)
    def forward(self, x):
        mask = self.activation(self.weight*self.attention(x))
        x = torch.cat([x, mask.expand(-1, self.inplanes, -1, -1)*x], 1)
        return self.out_activation(self.out(x))


class SqueezeNet(nn.Module):
    def __init__(self, version='1_1', num_classes=5):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.version = version
        final_conv = None
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256))
            # Final convolution is initialized differently from the rest
            final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
            self.masks = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv)
            self.attention = nn.Sigmoid()
            self.head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.LogSoftmax(dim=1))
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256))
            # Final convolution is initialized differently from the rest
            final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
            self.masks = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv)
            self.attention = nn.Sigmoid()
            self.head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.LogSoftmax(dim=1))
        elif version == 'FC':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256))
            # Final convolution is initialized differently from the rest
            final_fc = nn.Linear(512*13*13, self.num_classes)
            self.head = nn.Sequential(
                nn.Dropout(p=0.5),
                final_fc,
                nn.LogSoftmax(dim=1))
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0/1_1/FC expected".format(version=version))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        if self.version == 'FC':
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.head(x)
            return x, None
        else:
            x = self.features(x)
            x = self.masks(x)
            masks = self.attention(x)
            x = self.head(x)
            return x, masks


def squeezenet(version, snapshot=None, **kwargs):
    model = SqueezeNet(version, **kwargs)
    if snapshot is not None:
        model.load_state_dict(torch.load(snapshot), strict=False)
    return model