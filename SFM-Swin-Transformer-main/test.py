# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:19:48 2023

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pytorch_wavelets import DWT1DForward, DWT1DInverse,DWTForward, DWTInverse # or simply DWT1D, IDWT1D
import torch
from reformer_pytorch import LSHSelfAttention

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

def wavelet_function(x,dwtfunction,layer):
    result1 = []
    result1.append(x)
    result2 = result1
    num = 0 
    for i in range(layer):
        result1 = result2
        result2 = []
        for dwtx in result1:
            yl,yh =  dwtfunction[num](dwtx)
            result2.append(yl)
            result2.append(yh[0][:,:,0])
            result2.append(yh[0][:,:,1])
            result2.append(yh[0][:,:,2])
            num = num + 1
            # print(num)
        # print(len(result2))
    result = result2[0]
    for i in range(1,len(result2)):
        result = torch.cat((result,result2[i]),1)
    return result


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class Frequency_attention(nn.Module):
    def __init__(self, n_level = 1,wavelet_name="haar", input_channel = 3, output_channel = 12,emb_size = 256,input_resolution = None):
        super().__init__()
        self.n_level = n_level
        self.input_resolution = input_resolution
        self.spatial_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1).cuda()   
        self.input_channel =  input_channel
        self.output_channel = output_channel
        if n_level == 2:
            self.spatial_conv1 = nn.Conv2d(input_channel, output_channel, 8, stride=2, padding=3).cuda()
            
            
        self.dwtfunction = []
        for i in range(16):
            dwt = DWTForward(J=1, mode='periodization', wave=wavelet_name).cuda()
            self.dwtfunction.append(dwt)
        self.emb_size = emb_size
        self.num_heads = 8
        self.bucket_size = 64
        # self.mul_Attention = MultiHeadAttention(emb_size = emb_size).cuda() 
        self.mul_Attention = LSHSelfAttention(
            dim = emb_size,
            heads = self.num_heads,
            bucket_size = self.bucket_size,
            n_hashes = 8,
            causal = False).cuda()

        
    def forward(self, x):
        if self.n_level == 1:
            x_spatial = x
        else:
            x_spatial = self.spatial_conv1(x)
        
        x_spatial = self.spatial_pool(x_spatial)
        x_wavelet = wavelet_function(x,self.dwtfunction,self.n_level)
        x = torch.cat((x_spatial,x_wavelet),dim = 1)
        # print(x.shape)
        h = int(x.shape[2])
        x = rearrange(x, "b n h d -> b n (h d)") 
        # print(x.shape)
        x = self.mul_Attention(x)
        x = rearrange(x, "b n (h d) -> b n h d", h=h)     
        # print(x.shape)
        return x
    def flops(self):
        H, W = self.input_resolution
        input_channel = self.input_channel
        output_channel = self.output_channel
        kernel_size = 8 # for spatial_conv1
    
        # Spatial Convolution
        if self.n_level == 2:
            flops_conv = (H // 2) * (W // 2) * input_channel * output_channel * kernel_size**2
        else:
            flops_conv = 0
    
        # MultiHeadAttention
        # emb_size = self.emb_size
        # num_heads = self.num_heads
        # flops_mha = 3 * H * W * emb_size**2 + H * W * emb_size**2 + 2 * H * W * num_heads * (emb_size // num_heads)**2
        N = self.emb_size  # Assume emb_size is a rough estimate of sequence length for simplicity
        D = self.emb_size
        H = self.num_heads
        B = self.bucket_size  # Average bucket size

        # Hashing FLOPs - this is an approximation and might be different based on the hashing function
        flops_hashing = N * D
        
        # Attention calculation per bucket, considering Q * K.T multiplication, and Q @ V matrix multiplication
        flops_attention_per_bucket = 2 * B * B * D

        # Total attention FLOPs for all buckets
        flops_attention = flops_attention_per_bucket * (N / B)

        total_flops = flops_hashing + flops_attention
        total_flops = flops_conv + total_flops
        return total_flops
        
input_tensor = torch.randn(8, 3, 32, 32).cuda()


model = Frequency_attention(n_level=1, input_channel=3, output_channel=12,input_resolution = [32,32])

output = model(input_tensor)

print(model.flops())







        