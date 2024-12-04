import torch.nn as nn
import torch
from collections import OrderedDict
from models.vit_seg_modeling import LayerNorm,Block
import copy
import models.vit_seg_configs as configs
import math


class Conv2dReLU(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(Conv2dReLU, self).__init__()  # Call super().__init__()
        gn = 16 if out_channels%16 ==0 else out_channels//4
        self.conv =  nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels , kernel_size=kernel_size, stride=stride, bias=False, padding=padding)),
            ('gn', nn.GroupNorm(gn, out_channels, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

    def forward(self,x):
        return self.conv(x)
    
class Decoder(nn.Module):
    def __init__(self, img_size=256):
        super(Decoder, self).__init__()

        self.upsample = nn.Sequential(
                    Conv2dReLU(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1),
                    nn.UpsamplingBilinear2d(scale_factor=2)
        )
    
        self.body = nn.Sequential(
            OrderedDict([
                ('upblock1', nn.Sequential(
                    Conv2dReLU(in_channels=64+64, out_channels=64),
                    Conv2dReLU(in_channels=64, out_channels=64),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    Conv2dReLU(in_channels=64, out_channels=32)


                )),
                ('upblock2', nn.Sequential(
                    Conv2dReLU(in_channels=32+32, out_channels=32),
                    Conv2dReLU(in_channels=32, out_channels=32),
                )),
            
            ])
        )

        self.final_conv = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride=1,padding=0)

    def forward(self, x,features):

        x = self.upsample(x)
        i = 0
        
        # print(features)
        for block_name, block in self.body.named_children():
            # print(x.shape,features[i].shape)
            x = torch.cat([x, features[i]], dim=1)
            x = block(x)
            i +=1

        x = self.final_conv(x)

        return x

class Encoder(nn.Module):
    def __init__(self, img_size=256):
        super(Encoder, self).__init__()


        self.layers = nn.Sequential(
                OrderedDict([
                ('block1', nn.Sequential(
                    Conv2dReLU(in_channels=1, out_channels=32),
                    Conv2dReLU(in_channels=32, out_channels=32)
                )),
                ('block2', nn.Sequential(
                    Conv2dReLU(in_channels=32, out_channels=64),
                    Conv2dReLU(in_channels=64, out_channels=64)
                ))
            ])
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        
        # self.embedding = Conv2dReLU(in_channels=256,out_channels=60)


    def forward(self, x):
        # print(x.shape)
        features = []
        for block_name, block in self.layers.named_children():
            # print(x.shape)
            x = block(x)
            features.append(x)
            x = self.maxpool(x)
            # print(f'{block_name}: {x.shape}')

        # x = self.embedding(x)
        return x,features

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.encoder = Encoder()
    
        self.conv_layer = nn.Sequential(
            Conv2dReLU(in_channels=64,out_channels=128),
            Conv2dReLU(in_channels=128,out_channels=128)
        )
        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # if x.shape[1] == 1:
        #     x = x.repeat(1,3,1,1)
        x,features = self.encoder(x)
        # print("Encoder : ",x.shape)
        # for f in features:
        #     print(f.shape)
        x = self.conv_layer(x)
        x = self.decoder(x,features[::-1])
        x = self.sigmoid(x)

        return x