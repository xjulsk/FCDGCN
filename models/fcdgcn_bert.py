
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import DefaultDict
from einops import rearrange
import pywt

class CSA_ConvBlock(nn.Module):
    def __init__(self, c: int, use_spatial_conv: bool = True):
        super().__init__()
        self.c = c
        if use_spatial_conv:
            Conv = lambda: nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        else:
            Conv = lambda: nn.Conv2d(c, c, kernel_size=1, bias=False)

        self.fq = Conv()
        self.fk = Conv()
        self.fv = Conv()

        self.bn   = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=1, stride=1) 
        self.gamma = nn.Parameter(torch.zeros(1))  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        HW = H * W

        q = self.fq(x)
        k = self.fk(x)
        v = self.fv(x)

        q = q.view(B, C, HW)
        k = k.view(B, C, HW)
        v = v.view(B, C, HW)


        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)


        attn = torch.matmul(q, k.transpose(-2, -1)) / (HW ** 0.5)
        attn = F.softmax(attn, dim=-1)


        out = torch.matmul(attn, v).view(B, C, H, W)
        out = self.pool(out)
        out = x + self.gamma * out
        out = self.bn(out)
        out = self.relu(out)
        return out

def complex_softmax(z: torch.Tensor, dim: int = -1):

    real = F.softmax(z.real, dim=dim)
    imag = F.softmax(z.imag, dim=dim)
    return torch.complex(real, imag)

class FrequencyAttention(nn.Module):
    def __init__(self, in_dim: int, num_heads: int = 8, use_complex_attn: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.use_complex_attn = use_complex_attn

        self.q_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm_in = nn.BatchNorm2d(in_dim)
        self.norm_out = nn.BatchNorm2d(in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.proj_drop = nn.Dropout(0.0)
    

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        HW = H * W

        x_norm = self.norm_in(x)


        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        q_f = torch.fft.fft2(q)
        k_f = torch.fft.fft2(k)
        v_f = torch.fft.fft2(v)

        qh = rearrange(q_f, 'b (h c) hgt wdt -> b h c (hgt wdt)', h=self.num_heads)
        kh = rearrange(k_f, 'b (h c) hgt wdt -> b h c (hgt wdt)', h=self.num_heads)
        vh = rearrange(v_f, 'b (h c) hgt wdt -> b h c (hgt wdt)', h=self.num_heads)

        # Attention calculation
        attn = qh @ kh.transpose(-2, -1)
        attn = attn * self.temperature

        if self.use_complex_attn:
            attn = complex_softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn.real, dim=-1)

        attn = self.attn_drop(attn)

        out_f = attn @ vh
        out_f = rearrange(out_f, 'b h c (hgt wdt) -> b (h c) hgt wdt', hgt=H, wdt=W)

        out = torch.fft.ifft2(out_f).real


        out = x + self.gamma * out
        out = self.norm_out(out)

        return out

class CSA_FrequencyAttentionBlock(nn.Module):
    def __init__(self, high_dim, low_dim, num_heads=8, use_complex_attn=True, flag=True):
        super(CSA_FrequencyAttentionBlock, self).__init__()

        self.CSA = CSA_ConvBlock(high_dim)  
        self.FreqAttn = FrequencyAttention(
            in_dim=high_dim, 
            num_heads=num_heads,  
            use_complex_attn=use_complex_attn  
        )
        self.flag = flag

    def forward(self, x, x0):
        if self.flag:
            z = self.CSA(x) 
        else:
            z = x  
        z = self.FreqAttn(z) 
        return z


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FCDGCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(100, opt.polarities_dim)

    def forward(self, inputs):
        outputs1 = self.gcn_model(inputs)
        logits = self.classifier(outputs1)

        return logits, None


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, src_mask, aspect_mask, short_mask= inputs
        h = self.gcn(inputs)    
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)  
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, 100)  
        outputs1 = (h*aspect_mask).sum(dim=1) / asp_wn
        return outputs1   


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        self.attdim = 100
        self.W = nn.Linear(self.attdim, self.attdim)
        self.Wx = nn.Linear(self.attention_heads + self.attdim * 2, self.attention_heads)
        self.Wxx = nn.Linear(self.bert_dim, self.attdim)
        self.Wi = nn.Linear(self.attdim, 50)
        self.aggregate_W = nn.Linear(self.attdim * 2, self.attdim)

        self.attn = MultiHeadAttention(opt.attention_heads, self.attdim)
        
        self.csa_freq_inloop = nn.ModuleList([
            CSA_FrequencyAttentionBlock(self.attdim, self.attdim, num_heads=self.attention_heads)
            for _ in range(self.layers)
        ])
        
        self.csa_freq_post = CSA_FrequencyAttentionBlock(self.attdim, self.attdim, num_heads=self.attention_heads)
        self.gate_inloop = nn.Parameter(torch.ones(self.layers))
        self.gate_post = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, src_mask, aspect_mask, short_mask = inputs
        src_mask = src_mask.unsqueeze(-2)
        batch = src_mask.size(0)
        length = src_mask.size()[2]

        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        
        gcn_inputs = self.Wxx(gcn_inputs) # (B, T, 100)
        
        weight_adj = self.attn(gcn_inputs, gcn_inputs, short_mask, None, src_mask) # 简化了aspect传入
        gcn_outputs = gcn_inputs
        layer_list = [gcn_inputs]

        for i in range(self.layers):
            gcn_outputs_expanded = gcn_outputs.unsqueeze(1).expand(batch, self.attention_heads, length, self.attdim)
            Ax = torch.matmul(weight_adj, gcn_outputs_expanded)
            Ax = Ax.mean(dim=1)
            Ax = self.W(Ax)
            gcn_outputs = F.relu(Ax)

            x2d = gcn_outputs.transpose(1, 2).unsqueeze(2)
            y2d = self.csa_freq_inloop[i](x2d, x2d)
            gate = torch.sigmoid(self.gate_inloop[i])
            gcn_outputs = (1.0 - gate) * gcn_outputs + gate * y2d.squeeze(2).transpose(1, 2)
            # ----------------------------------------------------

            layer_list.append(gcn_outputs)
            gcn_outputs = self.gcn_drop(gcn_outputs) if i < self.layers - 1 else gcn_outputs

            weight_adj_p = weight_adj.permute(0, 2, 3, 1).contiguous()
            node_outputs1 = gcn_outputs.unsqueeze(1).expand(batch, length, length, self.attdim)
            node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
            node = torch.cat([node_outputs1, node_outputs2], dim=-1)
            edge_n = torch.cat([weight_adj_p, node], dim=-1)
            edge = self.Wx(edge_n)
            edge = self.gcn_drop(edge) if i < self.layers - 1 else edge
            weight_adj = edge.permute(0, 3, 1, 2).contiguous()

        x2d_post = gcn_outputs.transpose(1, 2).unsqueeze(2)
        y2d_post = self.csa_freq_post(x2d_post, x2d_post)
        gate_p = torch.sigmoid(self.gate_post)
        gcn_outputs = (1.0 - gate_p) * gcn_outputs + gate_p * y2d_post.squeeze(2).transpose(1, 2)

        node_outputs = F.relu(gcn_outputs)
        return node_outputs


def attention(query, key, short, aspect, weight_m, bias_m, mask=None, dropout=None):   
    d_k = query.size(-1)   
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    batch=len(scores)  
    p=weight_m.size(0)
    max=weight_m.size(1)
    weight_m=weight_m.unsqueeze(0).repeat(batch,1,1,1)

    aspect_scores = torch.tanh(torch.add(torch.matmul(aspect, key.transpose(-2, -1)), bias_m))  
    scores=torch.add(scores, aspect_scores)
    

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores=torch.add(scores, short)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

 
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()  
        self.d_k = d_model // h  
        self.h = h    
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k)) 
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.dense = nn.Linear(d_model, self.d_k)
    

    def forward(self, query, key, short, aspect, mask=None):   
        mask = mask[:, :, :query.size(1)]  
        if mask is not None:
            mask = mask.unsqueeze(1)  
        
        nbatches = query.size(0)  
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        
        batch, aspect_dim = aspect.size()[0], aspect.size()[1]
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)    
        aspect = self.dense(aspect) 
        aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size()[2], self.d_k)
        attn = attention(query, key,short,aspect, self.weight_m, self.bias_m, mask=mask, dropout=self.dropout)  
        return attn