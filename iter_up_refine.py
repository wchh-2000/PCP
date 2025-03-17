# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter
def conv5x5(in_dims, out_dims, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_dims, out_dims, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_dims, out_dims, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_dims, out_dims, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=stride, bias=False)
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)
class SpectralNorm(nn.Module):
    """
    Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
    and add _noupdate_u_v() for evaluation
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        # if torch.is_grad_enabled() and self.module.training:
        if self.module.training:
            self._update_u_v()
        else:
            self._noupdate_u_v()
        return self.module.forward(*args)
        
class BasicBlock(nn.Module):
    def __init__(self, indims, dims, upsample=False, norm_layer=None, large_kernel=False,spec_norm=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        conv = conv5x5 if large_kernel else conv3x3
        if spec_norm:
            self.conv1 = SpectralNorm(conv(indims, indims))
            self.conv2 = SpectralNorm(conv(indims, dims))
        else:
            self.conv1 = conv(indims, indims)
            self.conv2 = conv(indims, dims)
        self.bn1 = norm_layer(indims)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.bn2 = norm_layer(dims)
        self.upsample = upsample
        if self.upsample:
            self.upsample_layer=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if spec_norm:
                self.identity_layer = nn.Sequential(
                    SpectralNorm(conv1x1(indims, dims)),
                    norm_layer(dims),
                )
            else:
                self.identity_layer = nn.Sequential(
                    conv1x1(indims, dims),
                    norm_layer(dims),
                )

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample:#upsample时indims和dims不同，否则相同
            identity = self.identity_layer(identity)
        out += identity
        out = self.activation(out)

        return out
class OutBlock(nn.Module):
    def __init__(self, indim, outdim, kernel_size=3,
                norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(OutBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(indim, indim // 2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            norm_layer(indim // 2),
            activation,
            nn.Conv2d(indim // 2, outdim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
        )
    def forward(self, x):
        return self.block(x)
    
class IUR(nn.Module):
    # Iterative Upscaling Refinement Module
    def __init__(self, indim, layers, block=BasicBlock, norm_layer=None, large_kernel=False,spec_norm=True):
        #layers: block number in each layer
        super(IUR, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel

        self.indim = indim // 2

        self.layer1 = self._make_layer(block, 64, layers[0], spec_norm=spec_norm)
        self.layer2 = self._make_layer(block, 32, layers[1], spec_norm=spec_norm)
        self.layer3 = self._make_layer(block, 16, layers[2], spec_norm=spec_norm)
        
        self.refine4 = OutBlock(64,2)
        self.refine2 = OutBlock(32,2)
        self.refine1 = OutBlock(16,2)

        self.feat16_upscaling = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(indim, indim // 2, kernel_size=3, padding=1),
                norm_layer(indim // 2),
                nn.LeakyReLU(0.2, inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, dims, block_num, spec_norm=True):
        norm_layer = self._norm_layer

        layers = [block(self.indim + 5, dims, True, norm_layer, self.large_kernel,spec_norm)]#第一个block上采样 额外图像3通道+map2通道(mask+vmap)
        self.indim = dims
        for _ in range(1, block_num):
            layers.append(block(self.indim, dims, norm_layer=norm_layer, large_kernel=self.large_kernel,spec_norm=spec_norm))

        return nn.Sequential(*layers)

    def forward(self, feat16, img, map):#feat16为image_encoder输出特征 1/16原图，img为原图，map为归一化前结果, b,2,h/4,w/4
        size8=int(feat16.shape[2]*2)
        size8=(size8,size8)
        feat8 = self.feat16_upscaling(feat16)#b,256,h/16,w/16 -> b,128,h/8,w/8
        img8 = F.interpolate(img, size8, mode='bilinear', align_corners=True)
        map8 = F.interpolate(map, size8, mode='bilinear', align_corners=True)
        x = self.layer1(torch.cat((feat8, img8, map8), dim=1)) # b,128+5,h/8,w/8 -> b,64,h/4,w/4
        map4 = self.refine4(x)
        
        img4 = F.interpolate(img, x.shape[2:], mode='bilinear', align_corners=True)
        x = self.layer2(torch.cat((x, img4, map4), dim=1)) # b,32,h/2,w/2
        map2 = self.refine2(x)

        img2 = F.interpolate(img, x.shape[2:], mode='bilinear', align_corners=True)
        x = self.layer3(torch.cat((x, img2, map2), dim=1)) # b,16,h,w
        map1 = self.refine1(x) #b,2,h,w
        
        return dict(s1=map1,s2=map2,s4=map4) #1,1/2,1/4倍尺寸的map
if __name__ == '__main__':
    # from torchinfo import summary #pip install torchinfo

    model = IUR(256, [3, 3,2])#,large_kernel=True
    b=1
    size=512
    x = torch.randn(b, 256, size//16, size//16)
    img = torch.randn(b, 3, size, size)
    map = torch.randn(b, 2, size//4, size//4)#mask,vmap
    input = dict(feat16=x, img=img, map=map) 
    ret = model(x, img, map)
    # summary(model,input_data=input,depth=1)