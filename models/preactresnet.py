'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
from pickle import NONE
from numpy.core.numeric import flatnonzero
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import activation
from model_utils import Normalize
from model_utils import *

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=nn.ReLU(inplace=True)):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        self.activate_fun = activation

    def forward(self, x):
        out =self.activate_fun(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.activate_fun(self.bn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation=nn.ReLU(inplace=True)):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        self.activate_fun = activation

    def forward(self, x):
        out = self.activate_fun(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.activate_fun(self.bn2(out)))
        out = self.conv3(self.activate_fun(self.bn3(out)))
        out += shortcut
        return out


def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (PreActBlock, [2,2,2,2]),
        '34': (PreActBlock, [3,4,6,3]),
        '50': (PreActBottleneck, [3,4,6,3]),
        '101':(PreActBottleneck, [3,4,23,3]),
        '152':(PreActBottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class ResNet_New(nn.Module):
    def __init__(self,depth, num_classes=10, dataset='cifar10', use_FNandWN=False, activation='ReLU'):
        super(ResNet_New, self).__init__()
        self.activation = activation
        if activation == 'ReLU':
            self.activate_fun = nn.ReLU(inplace=True)
            print('ReLU')
        elif activation == 'Softplus':
            self.activate_fun = nn.Softplus(beta=10.0, threshold=20)
            print('Softplus')
        elif activation == 'GELU':
            self.activate_fun = nn.GELU()
            print('GELU')
        elif activation == 'ELU':
            self.activate_fun = nn.ELU(alpha=1.0, inplace=True)
            print('ELU')
        elif activation == 'LeakyReLU':
            self.activate_fun = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            print('LeakyReLU')
        elif activation == 'SELU':
            self.activate_fun = nn.SELU(inplace=True)
            print('SELU')
        elif activation == 'CELU':
            self.activate_fun = nn.CELU(alpha=1.2, inplace=True)
            print('CELU')
        elif activation == 'Tanh':
            self.activate_fun = nn.Tanh()
            print('Tanh')
        elif activation == 'Swish':
            self.activate_fun = Swish()
            print('Swish')
        else:
            self.activate_fun = nn.ReLU(inplace=True)
            print('ReLU')

        block, num_blocks = cfg(depth)
        self.final_dim = 512*block.expansion
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0],  stride=1, activation=self.activate_fun )
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, activation=self.activate_fun )
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, activation=self.activate_fun )
        self.layer4 = self._make_layer(block, self.final_dim, num_blocks[3], stride=2, activation=self.activate_fun )
        self.norm1_layer = Normalize(dataset)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)
 
        image_width = 512
        text_width= 50
        embed_dim = 64
        self.imag_proj = nn.Sequential(
            nn.Linear(image_width, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_width, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )


    def _make_layer(self, block, planes, num_blocks, stride, activation):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return mu + std * eps
 
    def extract_feat(self, x):
        out = self.norm1_layer(x)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.activate_fun(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        feat = out.view(out.size(0), -1)
        return feat

    def forward(self, x, flg=False, text_feature = None):
        feat = self.extract_feat(x)
        out = self.linear(feat)
        if flg:
            img_feat = self.imag_proj(feat)
            txt_feat = self.text_proj(text_feature)
            return out, img_feat, txt_feat
        else:
            return out

def get_ResNet_New(depth, num_classes=10, dataset='cifar10', use_FNandWN=False, activation='ReLU'):
    return ResNet_New(depth, num_classes, dataset, use_FNandWN=use_FNandWN, activation=activation)

def test():
    net = ResNet_New()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

# test()
