import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import numpy as np

# --- IR50 Backbone ---
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    return torch.div(input, torch.norm(input, 2, axis, True))

class BottleneckIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        self.shortcut_layer = nn.MaxPool2d(1, stride) if in_channel == depth else nn.Sequential(nn.Conv2d(in_channel, depth, 1, stride, bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, 3, 1, 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm2d(depth)
        )
    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)

class Backbone(nn.Module):
    def __init__(self, num_layers=50, drop_ratio=0.0):
        super(Backbone, self).__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        blocks = [
            [BottleneckIR(64, 64, 2), BottleneckIR(64, 64, 1), BottleneckIR(64, 64, 1)],
            [BottleneckIR(64, 128, 2), BottleneckIR(128, 128, 1), BottleneckIR(128, 128, 1), BottleneckIR(128, 128, 1)],
            [BottleneckIR(128, 256, 2)] + [BottleneckIR(256, 256, 1) for _ in range(13)],
            [BottleneckIR(256, 512, 2), BottleneckIR(512, 512, 1), BottleneckIR(512, 512, 1)]
        ]
        self.body = nn.Sequential(*[unit for block in blocks for unit in block])
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(drop_ratio), Flatten(), nn.Linear(512 * 7 * 7, 512), nn.BatchNorm1d(512))
    def forward(self, x):
        return l2_norm(self.body(self.input_layer(x)))

# --- MobileFaceNet Backbone ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        return self.bn(self.conv(x))

class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.conv = ConvBlock(in_c, groups, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_dw = ConvBlock(groups, groups, groups=groups, kernel=kernel, stride=stride, padding=padding)
        self.project = LinearBlock(groups, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.residual = residual
    def forward(self, x):
        out = self.project(self.conv_dw(self.conv(x)))
        return (x + out) if (self.residual and x.shape == out.shape) else out

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = nn.Sequential(*(DepthWise(64, 64, residual=True, groups=128) for _ in range(4)))
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = nn.Sequential(*(DepthWise(128, 128, residual=True, groups=256) for _ in range(6)))
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = nn.Sequential(*(DepthWise(128, 128, residual=True, groups=256) for _ in range(2)))
        self.conv_6_sep = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x):
        out = self.conv_5(self.conv_45(self.conv_4(self.conv_34(self.conv_3(self.conv_23(self.conv2_dw(self.conv1(x))))))))
        return self.conv_6_sep(out)

# --- GLAM & Attention Modules ---
class GLAM(nn.Module):
    def __init__(self, in_channels=512, num_reduced_channels=64, feature_map_size=4, kernel_size=3):
        super().__init__()
        self.lc_conv = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.gap = nn.AvgPool2d(feature_map_size)
        self.ls_1 = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.ls_2 = nn.Conv2d(int(num_reduced_channels*4), 1, 1, 1)
        self.ls_d3 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=1)
        self.ls_d5 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=2, dilation=2)
        self.ls_d7 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=3, dilation=3)
        
        self.gc_conv = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.gs_q = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.gs_k = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.gs_v = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.gs_att = nn.Conv2d(num_reduced_channels, in_channels, 1, 1)
        self.num_reduced_channels = num_reduced_channels
        self.fusion_weights = nn.Parameter(torch.Tensor([0.333, 0.333, 0.333]))

    def forward(self, x):
        N, C, H, W = x.shape
        lc_att = self.lc_conv(self.gap(x).reshape(N, 1, C)).sigmoid().reshape(N, C, 1, 1)
        lc_out = x * lc_att + x
        ls_feat = self.ls_1(x)
        ls_att = self.ls_2(torch.cat((ls_feat, self.ls_d3(ls_feat), self.ls_d5(ls_feat), self.ls_d7(ls_feat)), dim=1))
        local_att = (lc_out * ls_att + lc_out).unsqueeze(1)

        gc_q = self.gc_conv(self.gap(x).reshape(N, 1, C)).sigmoid()
        gc_key = self.gc_conv(self.gap(x).reshape(N, 1, C)).sigmoid().permute(0, 2, 1)
        gc_qk = torch.bmm(gc_key, gc_q).reshape(N, -1).softmax(-1).reshape(N, C, C)
        gc_out = x * torch.bmm(x.permute(0, 2, 3, 1).reshape(N, -1, C), gc_qk).permute(0, 2, 1).reshape(N, C, H, W)
        
        Cr = self.num_reduced_channels
        gs_q = self.gs_q(x).reshape(N, Cr, -1)
        gs_k = self.gs_k(x).reshape(N, Cr, -1)
        gs_qk = torch.bmm(gs_k.permute(0, 2, 1), gs_q).reshape(N, -1).softmax(-1).reshape(N, H*W, H*W)
        gs_att = self.gs_att(torch.bmm(self.gs_v(x).reshape(N, Cr, -1), gs_qk).reshape(N, Cr, H, W))
        global_att = (gc_out * gs_att + gc_out).unsqueeze(1)

        inputs_x = x.unsqueeze(1)
        weights = self.fusion_weights.softmax(-1).reshape(1, 3, 1, 1, 1)
        return torch.cat((local_att, inputs_x, global_att), dim=1).mul(weights).sum(1)

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    def forward(self, x):
        return self.attn(x, x, x)[0]

class PatchBasedUniqueAttention(nn.Module):
    def __init__(self, num_patches=49, dim=512):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_patches, dim))
    def forward(self, resnet_feat, mobile_feat):
        return resnet_feat + mobile_feat + self.weight.unsqueeze(0)

class TemporalConv(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).flatten(1)

class DFERmodel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.resnet = Backbone(50, 0.0)
        self.glam = GLAM(in_channels=512, num_reduced_channels=64, feature_map_size=4, kernel_size=3)
        self.face_landback = MobileFaceNet(512)
        self.patch_embed = PatchEmbedding()
        self.mhsa = MultiHeadSelfAttention()
        self.unique_attention = PatchBasedUniqueAttention(49, 512)
        self.ial_convtemp = TemporalConv(512)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x_resnet = self.resnet(x)
        x_resnet_glam = self.glam(x_resnet)
        x_mobilefacenet = self.face_landback(x)
        
        resnet_patches = self.patch_embed(x_resnet_glam)
        mobilefacenet_patches = self.patch_embed(x_mobilefacenet)
        
        resnet_mhsa = self.mhsa(resnet_patches)
        mobilefacenet_mhsa = self.mhsa(mobilefacenet_patches)
        
        fuse = self.unique_attention(resnet_mhsa, mobilefacenet_mhsa)
        frame_features = fuse.clone()
        
        _, npatch, embedding = fuse.shape
        fuse = fuse.reshape(b, t, npatch, embedding).permute(0, 2, 1, 3).reshape(b, npatch * t, embedding)
        x_temp = self.ial_convtemp(fuse)
        out = self.fc2(F.relu(self.fc1(x_temp)))
        return out, frame_features

class CombinedEmotionLoss(nn.Module):
    def __init__(self, class_weights=None, temp=2.0, alpha=0.05, beta=0.05, gamma=0.9, num_classes=6):
        super().__init__()
        self.base_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.temp = temp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prediction_memory = {}
        self.frame_classifier = nn.Linear(512, num_classes)
    
    def forward(self, frame_features, final_output, targets, sample_ids, b, t):
        if not hasattr(self, 'device_set'):
            self.frame_classifier = self.frame_classifier.to(frame_features.device)
            self.device_set = True
        
        hard_loss = self.base_criterion(final_output, targets)
        
        distillation_losses = []
        current_predictions = F.softmax(final_output / self.temp, dim=1)
        if sample_ids is not None:
            for idx, sample_id in enumerate(sample_ids):
                if sample_id in self.prediction_memory:
                    prev_pred = self.prediction_memory[sample_id]
                    dist_loss = F.kl_div(
                        F.log_softmax(final_output[idx:idx+1] / self.temp, dim=1),
                        prev_pred.unsqueeze(0),
                        reduction='batchmean'
                    ) * (self.temp ** 2)
                    distillation_losses.append(dist_loss)
                self.prediction_memory[sample_id] = current_predictions[idx].detach()
        
        distillation_loss = torch.stack(distillation_losses).mean() if distillation_losses else torch.tensor(0.0, device=final_output.device)
        
        feat_mean = frame_features.mean(dim=1)
        frame_logits = self.frame_classifier(feat_mean)
        frame_probs = F.softmax(frame_logits / self.temp, dim=1).view(b, t, -1)
        
        consistency_loss = torch.stack([F.mse_loss(video_frames[:-1], video_frames[1:]) for video_frames in frame_probs]).mean()
        voted_predictions = torch.stack([video_frames * F.softmax(video_frames.max(dim=1)[0], dim=0).unsqueeze(1) for video_frames in frame_probs]).sum(dim=1)
        
        total_loss = self.gamma * hard_loss + self.alpha * distillation_loss + self.beta * consistency_loss
        return total_loss, voted_predictions

    def reset_memory(self):
        self.prediction_memory.clear()