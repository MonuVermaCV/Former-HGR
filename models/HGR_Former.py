import torch
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math
import torchvision
from torchvision import datasets, models, transforms
from .ir50 import Backbone

#############################################################
def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        m=0.8
        w =torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w =torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.raw_weight=nn.Sigmoid()
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv3 = nn.Conv2d(hidden_features*2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv5 = nn.Conv2d(hidden_features*2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1 = self.dwconv3(x)
        x2 = self.dwconv5(x)
        weight =  self.raw_weight(self.w)
        # print(weight)
        
        
        x = weight.expand_as(x1)*x1+(1-weight.expand_as(x2)*x2)
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        # print(dim)
        # exit()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=16, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        #################weightoperation########
        m=0.7
        w =torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w =torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.raw_weight=nn.Sigmoid()
        #--------------------------
        # print(n_feat)
        self.first = nn.Conv2d(n_feat, n_feat//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.second =  nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False)
        

        self.third = nn.Conv2d(n_feat,n_feat*2 , kernel_size=3, stride=2, padding=1, bias=False)
        # self.pix = nn.PixelUnshuffle(2)

        # self.body = nn.Sequential()
                                  

    def forward(self, x):

        weight =  self.raw_weight(self.w)
       
        pix = self.third(x)
        return pix
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ResNet(nn.Module):   # S-Former after stage3

    def __init__(self, inp_channels=16, 
        out_channels=3,dim = 48,
        num_blocks = [2,4,4,6,6], 
        num_refinement_blocks = 4,
        heads = [1,2,4,6,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False):
        super(ResNet, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self. avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.down4_4 = Downsample(int(dim*2**3)) ## From Level 4 to Level 5
        ################
        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/ir50.pth', map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)
        
        self.BN1 = nn.BatchNorm2d(16)
        self.BN2 = nn.BatchNorm2d(256)
        self.conv_N1 = nn.Conv2d(3, 8, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_N2 = nn.Conv2d(3, 8, kernel_size=3, stride=1,padding=1, bias=False)
        self.conv_N3 = nn.Conv2d(3, 8, kernel_size=5, stride=1,padding=2, bias=False)
        self.ca = ChannelAttention(24)
        self.sa = SpatialAttention()
        self.conv_N4 = nn.Conv2d(3, 16, kernel_size=1, stride=2, padding=0, bias=False)
        self.conv_N5 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        m=0.8
        w =torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w =torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.raw_weight=nn.Sigmoid()
        
    def forward(self, x):
        
       
        x = x.contiguous().view(-1, 3, 112,112)
 
        x2 = self.conv_N5(x)
        out= x2

        #################
        x_ir = self.ir_back(x)

        x_ir = x_ir.view(-1, 16, 56,56)
        
        
        x_ir = out+x_ir
        
        
        ################
        
        inp_enc_level1 = self.patch_embed(x_ir)
        
        
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        
    

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        
        inp_enc_level4 = self.down3_4(out_enc_level3) 
        
        final = torch.flatten(inp_enc_level4, 1)

        
        return final


def spatial_transformer():
    return ResNet()


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = spatial_transformer()
    model(img)
