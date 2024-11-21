
import torch
import torch.nn as nn
import numpy as np

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding):
        super(depthwise_separable_conv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=self.kernel_size, padding=self.padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class conv_block_depthwise(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            depthwise_separable_conv(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            depthwise_separable_conv(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
        

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block_depthwise(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c)
        self.c1 = conv_block_depthwise(in_c[0]+out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x

class DS_attention_unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder_block(3, 8)
        self.e2 = encoder_block(8, 16)
        self.e3 = encoder_block(16, 32)
    
        self.b1 = conv_block_depthwise(32, 64)

        self.d1 = decoder_block([64, 32], 32)
        self.d2 = decoder_block([32, 16], 16)
        self.d3 = decoder_block([16, 8], 8)

        self.output = depthwise_separable_conv(8, 2, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        # print(s1.shape)
        s2, p2 = self.e2(p1)
        # print(s2.shape)
        s3, p3 = self.e3(p2)
        # print(s3.shape)

        b1 = self.b1(p3)
        # print(b1.shape)

        d1 = self.d1(b1, s3)
        # print(d1.shape)
        d2 = self.d2(d1, s2)
        # print(d2.shape)
        d3 = self.d3(d2, s1)
        # print(d3.shape)

        output = self.output(d3)
        return output

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

if __name__ == "__main__":
    x_half = torch.randn((1, 3, 96, 176))
    x = torch.randn((1, 3, 184, 320))
    model = DS_attention_unet()
    print("Total Param: ", netParams(model))
    output = model(x)
    print("full: ",output.shape)
    output = model(x_half)
    print("half: ",output.shape)