import torch
import torch.nn as nn
import torch.nn.functional as F

#定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        #这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)      
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
    #这个函数主要是用来，重复同一个残差块    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #在这里，整个ResNet18的结构就很清晰了
        #[1, 3, 256, 256]
        out = self.conv1(x)
        #[1, 64, 64, 64]
        out = self.layer1(out)
        #[1, 64, 64, 64]
        out = self.layer2(out)
        #[1, 128, 32, 32]
        out = self.layer3(out)
        #[1, 256, 16, 16]
        out = self.layer4(out)
        #[1, 512, 8, 8]
        out = self.avgpool(out)
        #[1, 512, 1, 1]
        out = out.view(out.size(0), -1)
        #[1, 512]
        out = self.fc(out)
        #[1, 2]
        return out


class TwowayResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=2):
        super(TwowayResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)  
        self.layer4 = self.make_layer_specific(ResBlock,512,512, 2, stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))        
        self.fc = nn.Linear(512, num_classes)
    #这个函数主要是用来，重复同一个残差块   
    def make_layer_specific(self, block,inchannel, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        channel_temp=inchannel
        for stride in strides:
            layers.append(block(channel_temp, channels, stride))
            channel_temp = channels
        return nn.Sequential(*layers)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward_oneway(self,x):
        #[1, 1, 256, 256]
        out = self.conv1(x)
        #[1, 64, 64, 64]
        out = self.layer1(out)
        #[1, 64, 64, 64]
        out = self.layer2(out)
        #[1, 128, 32, 32]
        out = self.layer3(out)
        #[1, 256, 16, 16]
        # out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out
    def forward(self, x):
        #[1, 3, 256, 256]
        # print(x.size())
        input1=x[:,0,:,:].unsqueeze(1)
        input2=x[:,1,:,:].unsqueeze(1)
        # .unsqueeze(0)
        #[1, 1, 256, 256]
        output1 = self.forward_oneway(input1)
        output2 = self.forward_oneway(input2)
        #[1, 256, 16, 16]
        out=torch.cat((output1,output2),dim=1)
        #[1, 512, 16, 16]
        out=self.layer4(out)
        #[1, 512, 8, 8]
        out = self.avgpool(out)
        #[1, 512, 1, 1]
        out = out.view(out.size(0), -1)
        #[1, 512]
        out= self.fc(out)
        #[1, 2]
        return out