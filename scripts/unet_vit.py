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
                    Conv2dReLU(in_channels=60,out_channels=64,kernel_size=3,stride=1,padding=1),
                    nn.UpsamplingBilinear2d(scale_factor=2)
        )
    
        self.body = nn.Sequential(
            OrderedDict([
                ('upblock1', nn.Sequential(
                    Conv2dReLU(in_channels=60+64, out_channels=64),
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
        # print("up:",x.shape)
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
                    Conv2dReLU(in_channels=64, out_channels=60)
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

class Transformer(nn.Module):
    def __init__(self, config, vis):
        super(Transformer, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
    
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet,self).__init__()
#         self.encoder = Encoder()
#         self.conv_layer = nn.Sequential(
#             Conv2dReLU(in_channels=64,out_channels=128),
#             Conv2dReLU(in_channels=128,out_channels=128)
#         )
#         self.decoder = Decoder()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self,x):
#         # if x.shape[1] == 1:
#         #     x = x.repeat(1,3,1,1)
#         x,features = self.encoder(x)

#         # print("Encoder : ",x.shape)
#         # for f in features:
#         #     print(f.shape)
#         x = self.conv_layer(x)
#         x = self.decoder(x,features[::-1])
#         x = self.sigmoid(x)

#         return x

class TUNet(nn.Module):
    def __init__(self,config,vis):
        super(TUNet,self).__init__()
        self.encoder = Encoder()
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride =2,padding=0)
        self.vit = Transformer(config,vis)
        # self.conv_layer = nn.Sequential(
        #     Conv2dReLU(in_channels=64,out_channels=128),
        #     Conv2dReLU(in_channels=128,out_channels=128)
        # )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1024, 60))
        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # if x.shape[1] == 1:
        #     x = x.repeat(1,3,1,1)
        x,features = self.encoder(x)
        x = self.maxpool(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2) 
        # print(x.shape)
        x = x+self.position_embeddings
        # print("After Position Embeddings : ",x.shape)
        x,attn_weights = self.vit(x)
        # print("After VIT:",x.shape)
        x = x.permute(0,2,1)
        bsz,n_patch,hidden = x.shape
        h= w = int(math.sqrt(hidden))
        x = x.contiguous().view(bsz,n_patch,h,w)
        # print("After Reshape L",x.shape)
        x = self.upsample(x)
        # print("After Upsample : ",x.shape)
        #     Conv2dReLU(in_channels=64,out_channels=128),
        #     Conv2dReLU(in_channels=128,out_channels=128)
        # )

        # print("Encoder : ",x.shape)
        # for f in features:
        #     print(f.shape)
        # x = self.conv_layer(x)
        x = self.decoder(x,features[::-1])
        x = self.sigmoid(x)

        return x

    
