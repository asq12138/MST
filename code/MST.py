import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import thop
from einops import rearrange, repeat
import time
__all__ = ["MST", "test"]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_cross_attention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm(x1), self.norm(x2), **kwargs)


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.swish = Swish()
        self.fc3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        f1 = x1 * self.swish(x2)
        f = self.fc3(f1)
        return f
    
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., talk_heads=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., talk_heads=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x1, x2):
        kv_1 = self.to_kv(x1).chunk(2, dim=-1)
        k_1, v_1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv_1)

        q_2 = self.to_q(x2)
        q_2 = rearrange(q_2, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q_2, k_1.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_1)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = out + x2
        return self.to_out(out)
    
class Triple_Transformer(nn.Module):
    def __init__(self, dim, depth_b1, depth_b2, depth_b3, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers_b1 = nn.ModuleList([])
        for _ in range(depth_b1):
            self.layers_b1.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim//heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

        self.layers_b2 = nn.ModuleList([])
        for _ in range(depth_b2):
            self.layers_b2.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim//heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

        self.layers_b3 = nn.ModuleList([])
        for _ in range(depth_b3):
            self.layers_b3.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim//heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ])) 

        self.cross_attn_1 = PreNorm_cross_attention(dim, Cross_Attention(dim, heads=heads, dim_head=dim//heads, dropout=dropout))
        self.cross_attn_2 = PreNorm_cross_attention(dim, Cross_Attention(dim, heads=heads, dim_head=dim//heads, dropout=dropout))
        self.cross_attn_3 = PreNorm_cross_attention(dim, Cross_Attention(dim, heads=heads, dim_head=dim//heads, dropout=dropout))

    def forward(self, x_branch1, x_branch2, x_branch3):

        for attn_1, ff_1 in self.layers_b1:
            x_branch1 = attn_1(x_branch1) + x_branch1
            x_branch1 = ff_1(x_branch1) + x_branch1
        
        for attn_2, ff_2 in self.layers_b2:
            x_branch2 = attn_2(x_branch2) + x_branch2
            x_branch2 = ff_2(x_branch2) + x_branch2

        for attn_3, ff_3 in self.layers_b3:
            x_branch3 = attn_3(x_branch3) + x_branch3
            x_branch3 = ff_3(x_branch3) + x_branch3 

        _, dim_1, _ = x_branch1.shape
        _, dim_2, _ = x_branch2.shape 
        _, dim_3, _ = x_branch3.shape   

        cls_1 = torch.narrow(x_branch1, 1, 0, 1)
        x_branch1_rest = torch.narrow(x_branch1, 1, 1, dim_1-1)
        cls_2 = torch.narrow(x_branch2, 1, 0, 1)
        x_branch2_rest = torch.narrow(x_branch2, 1, 1, dim_2-1)
        cls_3 = torch.narrow(x_branch3, 1, 0, 1)
        x_branch3_rest = torch.narrow(x_branch3, 1, 1, dim_3-1)

        cross_branch1 = torch.cat((cls_1,x_branch2_rest,x_branch3_rest),dim=1)
        code_cls1 = self.cross_attn_1(cross_branch1, cls_1)
        code_x1 = torch.cat((code_cls1,x_branch1_rest),dim=1)

        cross_branch2 = torch.cat((cls_2,x_branch1_rest,x_branch3_rest),dim=1)
        code_cls2 = self.cross_attn_2(cross_branch2, cls_2)
        code_x2 = torch.cat((code_cls2,x_branch2_rest),dim=1)

        cross_branch3 = torch.cat((cls_3,x_branch1_rest,x_branch2_rest),dim=1)
        code_cls3 = self.cross_attn_3(cross_branch3, cls_3)
        code_x3 = torch.cat((code_cls3,x_branch3_rest),dim=1)

        return code_x1, code_x2, code_x3
 
class GroupBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel, groups):
        super(GroupBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, in_channel, kernel_size=1, padding=0),
        )
        self.shortcut = nn.Sequential(
            nn.Identity(),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.shortcut(x)
        x3 = self.relu(x1 + x2)
        return x3

    def __call__(self, x):
        return self.forward(x)

class MST(nn.Module):
    def __init__(self, classes, dim=60):
        super(MST, self).__init__()
        self.pos_embedding_branch1 = nn.Parameter(torch.randn(1, 64, dim))
        self.cls_token_branch1 = nn.Parameter(torch.randn(1, 1, dim))
        self.We_branch1 = nn.Parameter(torch.randn(1, dim, dim))
        self.dropout_branch1 = nn.Dropout(0.1)

        self.pos_embedding_branch2 = nn.Parameter(torch.randn(1, 128, dim))
        self.cls_token_branch2 = nn.Parameter(torch.randn(1, 1, dim))
        self.We_branch2 = nn.Parameter(torch.randn(1, dim, dim))
        self.dropout_branch2 = nn.Dropout(0.1)

        self.pos_embedding_branch3 = nn.Parameter(torch.randn(1, 256, dim))
        self.cls_token_branch3 = nn.Parameter(torch.randn(1, 1, dim))
        self.We_branch3 = nn.Parameter(torch.randn(1, dim, dim))
        self.dropout_branch3 = nn.Dropout(0.1)
        # -----------------------------------------------------------------
        self.A_preconv1 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=32, stride=16),
            nn.ReLU()
        )
        self.A_preconv2 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=16, stride=8),
            nn.ReLU()
        )
        self.A_preconv3 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=8, stride=4),
            nn.ReLU()
        )
        # -----------------------------------------------------------------
        self.P_preconv1 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=32, stride=16),
            nn.ReLU()
        )
        self.P_preconv2 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=16, stride=8),
            nn.ReLU()
        )
        self.P_preconv3 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=8, stride=4),
            nn.ReLU()
        )
        # -----------------------------------------------------------------
        self.F_preconv1 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=32, stride=16),
            nn.ReLU()
        )
        self.F_preconv2 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=16, stride=8),
            nn.ReLU()
        )
        self.F_preconv3 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=8, stride=4),
            nn.ReLU()
        )
        # -----------------------------------------------------------------
        self.cov_1 = nn.Sequential(
            GroupBlock(64, 64, 8),
            GroupBlock(64, 64, 8)
        )
        self.cov_2 = nn.Sequential(
            GroupBlock(128, 128, 16),
            GroupBlock(128, 128, 16),
        )
        self.cov_3 = nn.Sequential(
            GroupBlock(256, 256, 32),
            GroupBlock(256, 256, 32),
        )
        self.coder_1 = Triple_Transformer(dim=dim, depth_b1=3, depth_b2=1, depth_b3=1,
                                           heads=4, mlp_dim=dim*2, dropout=0.3)
        self.coder_2 = Triple_Transformer(dim=dim, depth_b1=3, depth_b2=1, depth_b3=1,
                                           heads=4, mlp_dim=dim*2, dropout=0.3)
        self.coder_3 = Triple_Transformer(dim=dim, depth_b1=3, depth_b2=1, depth_b3=1,
                                           heads=4, mlp_dim=dim*2, dropout=0.3)
        self.head_mlp1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, classes),
        )
        self.head_mlp2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, classes),
        )
        self.head_mlp3 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, classes),
        )

    def forward(self, x):
        x = x.float()
        A1 = self.A_preconv1(x[:,0].reshape(x.shape[0],1,-1)).transpose(1, 2)
        P1 = self.P_preconv1(x[:,1].reshape(x.shape[0],1,-1)).transpose(1, 2)
        F1 = self.F_preconv1(x[:,2].reshape(x.shape[0],1,-1)).transpose(1, 2)
        x_branch1 = torch.cat((A1,P1,F1), dim=2)

        A2 = self.A_preconv2(x[:,0].reshape(x.shape[0],1,-1)).transpose(1, 2)       
        P2 = self.P_preconv2(x[:,1].reshape(x.shape[0],1,-1)).transpose(1, 2)
        F2 = self.F_preconv2(x[:,2].reshape(x.shape[0],1,-1)).transpose(1, 2)
        x_branch2 = torch.cat((A2,P2,F2), dim=2)
        
        A3 = self.A_preconv3(x[:,0].reshape(x.shape[0],1,-1)).transpose(1, 2)
        P3 = self.P_preconv3(x[:,1].reshape(x.shape[0],1,-1)).transpose(1, 2)             
        F3 = self.F_preconv3(x[:,2].reshape(x.shape[0],1,-1)).transpose(1, 2)
        x_branch3 = torch.cat((A3,P3,F3), dim=2)

        B, N, _ = x_branch1.shape
        # class token & position coding
        cls_tokens = repeat(self.cls_token_branch1, '1 n d -> b n d', b=B)  # (B, 1, D)
        x_branch1 = torch.cat((cls_tokens, torch.matmul(x_branch1, self.We_branch1)), dim=1)
        x_branch1 += self.pos_embedding_branch1[:, :(N + 1)]
        x_branch1 = self.dropout_branch1(x_branch1)
        x_branch1 = self.cov_1(x_branch1)

        B, N, _ = x_branch2.shape
        # class token & position coding
        cls_tokens = repeat(self.cls_token_branch2, '1 n d -> b n d', b=B)  # (B, 1, D)
        x_branch2 = torch.cat((cls_tokens, torch.matmul(x_branch2, self.We_branch2)), dim=1)
        x_branch2 += self.pos_embedding_branch2[:, :(N + 1)]
        x_branch2 = self.dropout_branch2(x_branch2)
        x_branch2 = self.cov_2(x_branch2)

        B, N, _ = x_branch3.shape
        # class token & position coding
        cls_tokens = repeat(self.cls_token_branch3, '1 n d -> b n d', b=B)  # (B, 1, D)
        x_branch3 = torch.cat((cls_tokens, torch.matmul(x_branch3, self.We_branch3)), dim=1)
        x_branch3 += self.pos_embedding_branch3[:, :(N + 1)]
        x_branch3 = self.dropout_branch3(x_branch3)
        x_branch3 = self.cov_3(x_branch3)

        # ----------------------------------------------------------------- Ttreble_Transformer
        code_x1, code_x2, code_x3 = self.coder_1(x_branch1, x_branch2, x_branch3)
        code_x1, code_x2, code_x3 = self.coder_2(code_x1, code_x2, code_x3)
        code_x1, code_x2, code_x3 = self.coder_3(code_x1, code_x2, code_x3)
   
        cls_1 = torch.narrow(code_x1, 1, 0, 1).flatten(1)
        cls_2 = torch.narrow(code_x2, 1, 0, 1).flatten(1)
        cls_3 = torch.narrow(code_x3, 1, 0, 1).flatten(1)

        ce_logits = torch.stack((self.head_mlp1(cls_1),self.head_mlp2(cls_2),self.head_mlp3(cls_3)), dim=0)
        ce_logits = torch.mean(ce_logits, dim=0)
        
        # return fc , self.fc[-1].weight 
        return ce_logits, None


def test():
    x = torch.randn(1, 3, 1024)
    net = MST(25)
    flops, params = thop.profile(net, inputs=(x,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")
    print("flops & params", [flops, params])
    print("--------")
    print("TransGroupNet : flops & params = 41.518M, 584.729K ")


if __name__ == "__main__":
    test()
