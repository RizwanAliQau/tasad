from email.mime import base
import torch 
import torch.nn as nn

class Seg_Network(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, base_width=128):
        super(Seg_Network, self).__init__()
        self.encoder = Encoder(in_channels, base_width)
        self.decoder = Decoder(base_width, out_channels=out_channels)

    def forward(self, x):
        b3 = self.encoder(x)
        output = self.decoder(b3)
        return output

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp


class Encoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super(Encoder, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width,kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
            )
        self.mp1    = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2,kernel_size=3,padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True)
            )
        self.mp2    = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4,kernel_size=3,padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True)
            )
        #self.mp3    = nn.Sequential(nn.MaxPool2d(2))
    
    def forward(self,x):
        b1          = self.block1(x)
        mp1         = self.mp1(b1)
        b2          = self.block2(mp1)
        mp2         = self.mp2(b2)
        b3          = self.block3(mp2)
        return b3


class Decoder(nn.Module):
    def __init__(self,base_width,out_channels=1):
        super(Decoder, self).__init__()

        self.up1    = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True)
            )
        
        self.db1    = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True)
            )

        self.up2    = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width*2,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True)
            )
        
        self.db2    = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
            )
        
        self.fin_out = nn.Sequential(nn.Conv2d(base_width,out_channels,kernel_size=3, padding=1)) #nn.Sigmoid(inplace=True)
    
    def forward(self, b3):
        up1         = self.up1(b3)
        db1         = self.db1(up1)
        up2         = self.up2(db1)
        db2         = self.db2(up2)

        out         = self.fin_out(db2)

        return out



