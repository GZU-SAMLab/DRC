import torch.nn as nn
import torch
import torch.nn.functional as F

def similarity_batch_count(sequences,sets_topN):
    sequences=sequences.reshape(len(sequences),1,sequences.shape[1],1)
    sets_topN=sets_topN.reshape(1,len(sets_topN),1,sets_topN.shape[1])
    # v shape : [sequences.shape[0],set_topN.shape[0],sequences.shape[1],set_topN.shape[1]]
    v = (~((sequences+1) / (sets_topN+1) -1).bool()).float()
  #  l=torch.tensor(np.arange(1,lower_limit,-(1-lower_limit)/sequences.shape[2])).to(device)
    # diag(l) shape: [1,1,sequences.shape[1],sequences.shape[1]]
    # matrix shape: [sequences.shape[0],set_topN.shape[0],sequence.shape[1],set_topN.shape[1]]
#    matrix=torch.matmul(torch.diag(l).unsqueeze(0).unsqueeze(0).float(),v.float())
    # value_topN shape: [1,1,set_topN.shape[1],set_topN.shape[1]]
 #   value_topN=torch.tensor(np.arange(1,lower_limit,-(1-lower_limit)/sets_topN.shape[3])).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,1,sets_topN.shape[3],1).to(device)
    ###a=torch.zeros(1,1,sets_topN.shape[3],1).to(device)
    ###value_topN=value_topN+a
    # output shape: [sequences.shape[0],set_topN.shape[0],sequence.shape[1],set_topN.shape[1]]
 #   output=torch.matmul(matrix,value_topN.float()).sum([2,3])
    return v.sum([2,3])

def leaky_tanh(x, threshold, alpha=0.05):
    x = x - threshold
    x = x * (8/0.2)
    return torch.max(alpha*x, torch.tanh(x))

def leaky_tanh1(x, length_threshold, alpha, m):
    threshold = torch.sort(x, descending=True).values[:, length_threshold].unsqueeze(1)
    x = x - threshold
    ratio = m / torch.max(x, dim=1).values.unsqueeze(1)
    x = x * ratio
    return torch.max(alpha*x, torch.tanh(x))



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64
                 ):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.n=512
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            #self.fc=nn.Linear(self.n,5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def mk_layer(self):
        self.fc1=nn.Linear(200,5)


    def forward(self, x, proxies=None, length_threshold=70, m=3, alpha=0.01):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        if self.bce:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = F.normalize(x, dim=1, p=2)
            y = leaky_tanh1(x, length_threshold = length_threshold, m=m, alpha=alpha)
            y = torch.matmul(y, proxies.transpose(0,1))
            return x,y

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152(num_classes=1000, include_top=True):
    # "https://download.pytorch.org/models/resnet152-f82ba261.pth"
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
