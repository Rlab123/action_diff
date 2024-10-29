# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
import math
import torch
import torch.nn as nn
import copy
import random as rad
import numpy as np
import time as Time

import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)
        if not isinstance(m.bias, type(None)):
            m.bias.data.fill_(0)



def get_timestep_embedding(timesteps, embedding_dim):  # for diffusion model
    # timesteps: batch,
    # out:       batch, embedding_dim
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def swish(x):
    return x * torch.sigmoid(x)


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def normalize(x, scale):  # [0,1] > [-scale, scale]
    x = (x * 2 - 1.) * scale
    return x


def denormalize(x, scale):  # [-scale, scale] > [0,1]
    x = ((x / scale) + 1) / 2
    return x


# ref: Weakly-supervised Temporal Action Localization with Multi-head Cross-modal Attention (PRICAI 2022)
class MCA(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super(MCA, self).__init__()
        self.rgb_proj = nn.Parameter(torch.empty(num_head, feat_dim, feat_dim // num_head))
        self.flow_proj = nn.Parameter(torch.empty(num_head, feat_dim, feat_dim // num_head))
        self.atte = nn.Parameter(torch.empty(num_head, feat_dim // num_head, feat_dim // num_head))

        nn.init.uniform_(self.rgb_proj, -math.sqrt(feat_dim), math.sqrt(feat_dim))
        nn.init.uniform_(self.flow_proj, -math.sqrt(feat_dim), math.sqrt(feat_dim))
        nn.init.uniform_(self.atte, -math.sqrt(feat_dim // num_head), math.sqrt(feat_dim // num_head))
        self.num_head = num_head

    def forward(self, rgb, flow):
        rgb, flow = rgb.transpose(-1, -2).contiguous(), flow.transpose(-1, -2).contiguous()
        n, t, d = rgb.shape
        # [N, H, T, D/H]
        o_rgb = F.normalize(torch.matmul(rgb.unsqueeze(dim=1), self.rgb_proj), dim=-1)
        o_flow = F.normalize(torch.matmul(flow.unsqueeze(dim=1), self.flow_proj), dim=-1)
        # [N, H, T, T]
        atte = torch.matmul(torch.matmul(o_rgb, self.atte), o_flow.transpose(-1, -2).contiguous())
        rgb_atte = torch.softmax(atte, dim=-1)
        flow_atte = torch.softmax(atte.transpose(-1, -2).contiguous(), dim=-1)

        # [N, H, T, D/H]
        e_rgb = F.gelu(torch.matmul(rgb_atte, o_rgb))
        e_flow = F.gelu(torch.matmul(flow_atte, o_flow))
        # [N, T, D]
        f_rgb = torch.tanh(e_rgb.transpose(-1, -2).reshape(n, t, -1).contiguous() + rgb)
        f_flow = torch.tanh(e_flow.transpose(-1, -2).reshape(n, t, -1).contiguous() + flow)

        f_rgb, f_flow = f_rgb.transpose(-1, -2).contiguous(), f_flow.transpose(-1, -2).contiguous()
        return f_rgb, f_flow


class TFE_DC_Module(nn.Module):
    def __init__(self, n_feature):
        super().__init__()

        embed_dim = 1024
        self.layer1 = nn.Sequential(nn.Conv1d(n_feature, embed_dim, 3, padding=2 ** 0, dilation=2 ** 0),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))
        self.layer2 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=2 ** 1, dilation=2 ** 1),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))
        self.layer3 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=2 ** 2, dilation=2 ** 2),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))

        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())

    def forward(self, x):
        out = self.layer1(x)
        out_attention1 = self.attention(torch.sigmoid(out) * x)

        out = self.layer2(out)
        out_attention2 = self.attention(torch.sigmoid(out) * x)

        out = self.layer3(out)
        out_feature = torch.sigmoid(out) * x
        out_attention3 = self.attention(out_feature)

        out_attention = (out_attention1 + out_attention2 + out_attention3) / 3.0

        return out_attention.transpose(-1, -2), out_feature


class BWA(torch.nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        embed_dim = 1024
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, (3,), padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, (3,), padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv1d(512, 1, (1,)),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        self.channel_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, vfeat, ffeat):
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)
        bit_wise_attn = self.bit_wise_attn(ffeat)
        filter_feat = torch.sigmoid(bit_wise_attn * channel_attn) * vfeat
        x_atn = self.attention(filter_feat)
        return x_atn.transpose(-1, -2), filter_feat


class Model(nn.Module):
    def __init__(self, num_classes,diffusion_params, decoder_params,device):
        super(Model, self).__init__()

        self.mca = MCA(1024)
        self.rgb_bwa = BWA(1024)
        self.flow_bwa = BWA(1024)
        self.fusion = 0.7
        self.beta = 0.15
        self.intervel = 4

        # ref: A Hybrid Attention Mechanism for Weakly-Supervised Temporal Action Localization (AAAI 2021)
        if num_classes != 20:
            self.cas_rgb_encoder = nn.Sequential(nn.Conv1d(1024, 1024, 3, padding=1), nn.ReLU(),
                                                 nn.Conv1d(1024, num_classes, kernel_size=1),
                                                 nn.AvgPool1d(13, 1, padding=6, count_include_pad=True))
            self.cas_flow_encoder = nn.Sequential(nn.Conv1d(1024, 1024, 3, padding=1), nn.ReLU(),
                                                  nn.Conv1d(1024, num_classes, kernel_size=1),
                                                  nn.AvgPool1d(13, 1, padding=6, count_include_pad=True))

            self.aas_rgb_encoder = nn.Sequential(nn.Conv1d(1024, 512, 1),
                                                 nn.ReLU(), nn.Conv1d(512, 1, 1),
                                                 nn.AvgPool1d(13, 1, padding=6, count_include_pad=True))
            self.aas_flow_encoder = nn.Sequential(nn.Conv1d(1024, 512, 1),
                                                  nn.ReLU(), nn.Conv1d(512, 1, 1),
                                                  nn.AvgPool1d(13, 1, padding=6, count_include_pad=True))

            self.channel_avg = nn.AvgPool1d(kernel_size=2)
        else:
            self.cas_rgb_encoder = nn.Sequential(nn.Conv1d(1024, 1024, 3, padding=1), nn.ReLU(),
                                                 nn.Conv1d(1024, num_classes, kernel_size=1))
            self.cas_flow_encoder = nn.Sequential(nn.Conv1d(1024, 1024, 3, padding=1), nn.ReLU(),
                                                  nn.Conv1d(1024, num_classes, kernel_size=1))

            self.aas_rgb_encoder = nn.Sequential(nn.Conv1d(1024, 512, 1), nn.ReLU(), nn.Conv1d(512, 1, 1))
            self.aas_flow_encoder = nn.Sequential(nn.Conv1d(1024, 512, 1), nn.ReLU(), nn.Conv1d(512, 1, 1))

        #self.apply(weights_init)

        self.device=device
        self.num_classes=num_classes
        timesteps = diffusion_params['timesteps']
        betas = cosine_beta_schedule(timesteps)  # torch.Size([1000])
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = diffusion_params['sampling_timesteps']
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = diffusion_params['ddim_sampling_eta']
        self.scale = diffusion_params['snr_scale']

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        decoder_params['num_classes'] = num_classes
        decoder_params['input_dim'] = 100

        self.detach_decoder = diffusion_params['detach_decoder']
        self.cond_types = diffusion_params['cond_types']

        self.decoder = DecoderModel(**decoder_params)
        self.apply(weights_init)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise=None):  # forward diffusion
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def model_predictions(self, backbone_feats, x, t):

        x_m = torch.clamp(x, min=-1 * self.scale, max=self.scale)  # [-scale, +scale]
        x_m = denormalize(x_m, self.scale)  # [0, 1]

        assert (x_m.max() <= 1 and x_m.min() >= 0)
        x_start = self.decoder(backbone_feats, t, x_m.float())  # torch.Size([1, C, T])
        x_start = F.softmax(x_start, 1)
        assert (x_start.max() <= 1 and x_start.min() >= 0)

        x_start = normalize(x_start, self.scale)  # [-scale, +scale]
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)

        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def prepare_targets(self, event_gt):

        # event_gt: normalized [0, 1]

        assert (event_gt.max() <= 1 and event_gt.min() >= 0)

        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()

        noise = torch.randn(size=event_gt.shape, device=self.device)

        x_start = (event_gt * 2. - 1.) * self.scale  # [-scale, +scale]

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        event_diffused = ((x / self.scale) + 1) / 2.  # normalized [0, 1]

        return event_diffused, noise, t

    @torch.no_grad()
    def ddim_sample_my(self, backbone_feats, x_time, seed=None):

        # if self.use_instance_norm:
        #    video_feats = self.ins_norm(video_feats)

        # encoder_out, backbone_feats = self.encoder(video_feats, get_features=True)

        if seed is not None:
            rad.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # torch.Size([1, 19, 4847])
        shape = (backbone_feats.shape[0], self.num_classes, backbone_feats.shape[2])
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        # print("sampling_timesteps=",sampling_timesteps)
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        # tensor([ -1., 249., 499., 749., 999.])
        times = list(reversed(times.int().tolist()))
        # print("times=", times)
        # [[999, 959, 919, 879, 839, -1]
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # [(999, 749), (749, 499), (499, 249), (249, -1)]
        # print("time_pairs=", time_pairs)
        # x_time = torch.randn(shape, device=self.device)

        x_start = None
        for time, time_next in time_pairs:

            time_cond = torch.full((1,), time, device=self.device, dtype=torch.long)

            pred_noise, x_start = self.model_predictions(backbone_feats, x_time, time_cond)

            x_return = torch.clone(x_start)

            if time_next < 0:
                x_time = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_time)

            x_time = x_start * alpha_next.sqrt() + \
                     c * pred_noise + \
                     sigma * noise

        x_return = denormalize(x_return, self.scale)

        if seed is not None:
            t = 1000 * Time.time()  # current time in milliseconds
            t = int(t) % 2 ** 16
            rad.seed(t)
            torch.manual_seed(t)
            torch.cuda.manual_seed_all(t)

        return x_return

    @torch.no_grad()
    def ddim_sample(self, backbone_feats, seed=None):

        # if self.use_instance_norm:
        #    video_feats = self.ins_norm(video_feats)

        # encoder_out, backbone_feats = self.encoder(video_feats, get_features=True)

        if seed is not None:
            rad.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # torch.Size([1, 19, 4847])
        shape = (backbone_feats.shape[0], self.num_classes, backbone_feats.shape[2])
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        # print("sampling_timesteps=",sampling_timesteps)
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        # tensor([ -1., 249., 499., 749., 999.])
        times = list(reversed(times.int().tolist()))
        # print("times=", times)
        # [[999, 959, 919, 879, 839, -1]
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # [(999, 749), (749, 499), (499, 249), (249, -1)]
        # print("time_pairs=", time_pairs)
        x_time = torch.randn(shape, device=self.device)

        x_start = None
        for time, time_next in time_pairs:

            time_cond = torch.full((1,), time, device=self.device, dtype=torch.long)

            pred_noise, x_start = self.model_predictions(backbone_feats, x_time, time_cond)

            x_return = torch.clone(x_start)

            if time_next < 0:
                x_time = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_time)

            x_time = x_start * alpha_next.sqrt() + \
                     c * pred_noise + \
                     sigma * noise

        x_return = denormalize(x_return, self.scale)

        if seed is not None:
            t = 1000 * Time.time()  # current time in milliseconds
            t = int(t) % 2 ** 16
            rad.seed(t)
            torch.manual_seed(t)
            torch.cuda.manual_seed_all(t)

        return x_return

    def infer(self, x):
        # [N, D, T]

        x = x.transpose(-1, -2).contiguous()
        rgb0, flow0 = x[:, :1024, :], x[:, 1024:, :]
        rgb1, flow1 = self.mca(rgb0, flow0)
        atn_rgb, rgb = self.rgb_bwa(rgb1, flow1)
        atn_flow, flow = self.flow_bwa(flow1, rgb1)
        # [N, T, C], class activation sequence
        cas_rgb = self.cas_rgb_encoder(rgb).transpose(-1, -2).contiguous()
        cas_flow = self.cas_flow_encoder(flow).transpose(-1, -2).contiguous()

        # cas_score = torch.softmax(cas, dim=-1)
        # [N, T, 1], action activation sequence
        aas_rgb = torch.sigmoid(self.aas_rgb_encoder(rgb).transpose(-1, -2).contiguous())
        aas_flow = torch.sigmoid(self.aas_flow_encoder(flow).transpose(-1, -2).contiguous())

        aas_score = (aas_rgb + aas_flow + atn_rgb + atn_flow) / 4

        # [N, T, C]
        cas = cas_rgb + cas_flow
        cas_score = torch.softmax(cas, dim=-1)
        seg_score = (cas_score + aas_score) / 2
        n, t, c = seg_score.shape

        th_t = int(t / self.intervel)
        seg_mask_list = []

        if self.intervel < seg_score.shape[1]:
            for i in range(self.intervel - 1):
                seg_score_i = seg_score[:, th_t * i:th_t * (i + 1), :]
                cas_i = cas[:, th_t * i:th_t * (i + 1), :]
                seg_mask_i = temporal_clustering(seg_score_i)

                seg_mask_i = mask_refining(seg_score_i, seg_mask_i, cas_i)
                seg_mask_list.append(seg_mask_i)

            seg_score_i = seg_score[:, th_t * (self.intervel - 1):, :]
            cas_i = cas[:, th_t * (self.intervel - 1):, :]
            seg_mask_i = temporal_clustering(seg_score_i)

            seg_mask_i = mask_refining(seg_score_i, seg_mask_i, cas_i)
            seg_mask_list.append(seg_mask_i)
        else:

            seg_mask_i = temporal_clustering(seg_score)

            seg_mask_i = mask_refining(seg_score, seg_mask_i, cas)
            seg_mask_list.append(seg_mask_i)

        seg_mask = torch.cat(seg_mask_list, dim=1)

        # [N, C]
        #act_score, bkg_score = calculate_score(seg_score, seg_mask, cas)

        event_out = self.ddim_sample(cas.transpose(-1, -2))

        # print("event_out=", event_out.shape)
        event_out = event_out.transpose(-1, -2)

        cas1 = cas  + event_out
        cas_score1 = torch.softmax(cas1, dim=-1)
        act_score, bkg_score2 = calculate_score(seg_score, seg_mask, cas1 / 2)

        return act_score, seg_score

    def cas_attention(self, x):
        # [N, D, T]

        x = x.transpose(-1, -2).contiguous()
        rgb0, flow0 = x[:, :1024, :], x[:, 1024:, :]
        rgb1, flow1 = self.mca(rgb0, flow0)
        atn_rgb, rgb = self.rgb_bwa(rgb1, flow1)
        atn_flow, flow = self.flow_bwa(flow1, rgb1)
        # [N, T, C], class activation sequence
        cas_rgb = self.cas_rgb_encoder(rgb).transpose(-1, -2).contiguous()
        cas_flow = self.cas_flow_encoder(flow).transpose(-1, -2).contiguous()

        # cas_score = torch.softmax(cas, dim=-1)
        # [N, T, 1], action activation sequence
        aas_rgb = torch.sigmoid(self.aas_rgb_encoder(rgb).transpose(-1, -2).contiguous())
        aas_flow = torch.sigmoid(self.aas_flow_encoder(flow).transpose(-1, -2).contiguous())

        atn_rgb0 = 0
        atn_flow0 = 0
        aas_score = (aas_rgb + aas_flow + atn_rgb + atn_flow) / 4

        # aas_score = (aas_rgb + aas_flow + atn_rgb + atn_flow + atn_rgb0 + atn_flow0) / 6
        # [N, T, C]
        cas = cas_rgb + cas_flow
        cas_score = torch.softmax(cas, dim=-1)
        seg_score = (cas_score + aas_score) / 2

        return seg_score, aas_rgb,aas_flow, atn_rgb, \
               atn_flow, atn_rgb0, atn_flow0, cas


    def local_mask(self,seg_score,cas):
        n, t, c = seg_score.shape

        th_t = int(t / self.intervel)
        seg_mask_list = []

        for i in range(self.intervel - 1):
            seg_score_i = seg_score[:, th_t * i:th_t * (i + 1), :]
            cas_i = cas[:, th_t * i:th_t * (i + 1), :]
            seg_mask_i = temporal_clustering(seg_score_i)

            seg_mask_i = mask_refining(seg_score_i, seg_mask_i, cas_i)
            seg_mask_list.append(seg_mask_i)

        seg_score_i = seg_score[:, th_t * (self.intervel - 1):, :]
        cas_i = cas[:, th_t * (self.intervel - 1):, :]
        seg_mask_i = temporal_clustering(seg_score_i)
        seg_mask_i = mask_refining(seg_score_i, seg_mask_i, cas_i)

        seg_mask_list.append(seg_mask_i)

        seg_mask = torch.cat(seg_mask_list, dim=1)
        return seg_mask

    def diffusion_pro(self, backbone_feats, seg_mask):  # only for train

        if self.detach_decoder:
            backbone_feats = backbone_feats.detach()

        event_diffused, noise, t = self.prepare_targets(seg_mask)
        assert (event_diffused.max() <= 1 and event_diffused.min() >= 0)

        event_out = self.decoder(backbone_feats, t, event_diffused.float())
        event_out = event_out.transpose(-1, -2)
        return event_out

def temporal_clustering(seg_score):
    n, t, c = seg_score.shape
    # [N*C, T]
    seg_score = seg_score.transpose(-1, -2).contiguous().view(-1, t)
    sort_value, sort_index = torch.sort(seg_score, dim=-1, descending=True)
    mask = torch.zeros_like(seg_score)
    row_index = torch.arange(mask.shape[0], device=mask.device)
    # the index of the largest value is inited as positive
    mask[row_index, sort_index[:, 0]] = 1
    # [N*C]
    pos_sum, neg_sum = sort_value[:, 0], sort_value[:, -1]
    pos_num, neg_num = torch.ones_like(pos_sum), torch.ones_like(neg_sum)
    for i in range(1, t - 1):
        pos_center = pos_sum / pos_num
        neg_center = neg_sum / neg_num
        index, value = sort_index[:, i], sort_value[:, i]
        pos_distance = torch.abs(value - pos_center)
        neg_distance = torch.abs(value - neg_center)
        condition = torch.le(pos_distance, neg_distance)
        pos_list = torch.where(condition, value, torch.zeros_like(value))
        neg_list = torch.where(~condition, value, torch.zeros_like(value))
        # update centers
        pos_num = pos_num + condition.float()  # / (i + 1)
        pos_sum = pos_sum + pos_list  # / (i + 1)
        neg_num = neg_num + (~condition).float()
        neg_sum = neg_sum + neg_list
        # update mask
        mask[row_index, index] = condition.float()
    # [N, T, C]
    mask = mask.view(n, c, t).transpose(-1, -2).contiguous()
    return mask


def mask_refining(seg_score, seg_mask, cas):
    n, t, c = seg_score.shape
    seg_score = seg_score.transpose(-1, -2).contiguous().view(-1, t)

    sort_value, sort_index = torch.sort(seg_score, dim=-1, descending=True)

    seg_mask = seg_mask.transpose(-1, -2).contiguous().view(-1, t)

    row_index = torch.arange(seg_mask.shape[0], device=seg_mask.device).view(-1, 1).expand(-1, t).contiguous()

    sort_mask = seg_mask[row_index, sort_index]
    cas = cas.transpose(-1, -2).contiguous().view(-1, t)
    sort_cas = cas[row_index, sort_index]
    # [1, T]
    rank = torch.arange(2, t + 2, device=seg_score.device).float().unsqueeze(dim=0).reciprocal()
    # rank=torch.ones(t, device=seg_score.device).float().unsqueeze(dim=0)
    # [N*C]
    act_num = (rank * sort_mask).sum(dim=-1)

    act_score = (sort_cas * rank * sort_mask).sum(dim=-1) / torch.clamp_min(act_num, 1.0)

    act_score = torch.unsqueeze(act_score, dim=1)

    tmp_mask = torch.ge(sort_cas, act_score).long()
    t_m = tmp_mask.sum(dim=-1) - 1
    tp = sort_value.shape[1] - 1
    t_m[t_m > tp] = tp
    t_m[t_m < 0] = 0
    t_m = torch.unsqueeze(t_m, dim=1)

    mask = torch.zeros_like(sort_value).long()
    mask = mask.scatter_(1, t_m, torch.ones_like(t_m))
    mean_score = sort_value[mask.bool()]

    max_mask = torch.ge(seg_score, mean_score.unsqueeze(dim=1)).float()
    refined_mask = seg_mask * max_mask
    refined_mask = refined_mask.view(n, c, t).transpose(-1, -2)
    return refined_mask


def calculate_score(seg_score, seg_mask, cas):
    n, t, c = seg_score.shape
    # [N*C, T]
    seg_score = seg_score.transpose(-1, -2).contiguous().view(-1, t)
    sort_value, sort_index = torch.sort(seg_score, dim=-1, descending=True)
    seg_mask = seg_mask.transpose(-1, -2).contiguous().view(-1, t)
    row_index = torch.arange(seg_mask.shape[0], device=seg_mask.device).view(-1, 1).expand(-1, t).contiguous()
    sort_mask = seg_mask[row_index, sort_index]
    cas = cas.transpose(-1, -2).contiguous().view(-1, t)
    sort_cas = cas[row_index, sort_index]
    # [1, T]
    rank = torch.arange(2, t + 2, device=seg_score.device).float().unsqueeze(dim=0).reciprocal()
    # rank = torch.ones(t, device=seg_score.device).float().unsqueeze(dim=0)
    # [N*C]
    act_num = (rank * sort_mask).sum(dim=-1)
    act_score = (sort_cas * rank * sort_mask).sum(dim=-1) / torch.clamp_min(act_num, 1.0)
    bkg_num = (1.0 - sort_mask).sum(dim=-1)
    bkg_score = (sort_cas * (1.0 - sort_mask)).sum(dim=-1) / torch.clamp_min(bkg_num, 1.0)
    act_score, bkg_score = torch.softmax(act_score.view(n, c), dim=-1), torch.softmax(bkg_score.view(n, c), dim=-1)
    return act_score, bkg_score


def cross_entropy(act_score, bkg_score, label, eps=1e-8):
    act_num = torch.clamp_min(torch.sum(label, dim=-1), 1.0)
    act_loss = (-(label * torch.log(torch.clamp_min(act_score, eps))).sum(dim=-1) / act_num).mean()
    bkg_loss = (-torch.log(torch.clamp_min(1.0 - bkg_score, eps))).mean()
    return act_loss + bkg_loss


# ref: Weakly Supervised Action Selection Learning in Video (CVPR 2021)
def generalized_cross_entropy(aas_score, seg_mask, label, q=0.7, eps=1e-8):
    # [N, T]
    aas_score = aas_score.squeeze(dim=-1)
    n, t, c = seg_mask.shape
    # [N, T]
    mask = torch.zeros(n, t, device=seg_mask.device)
    for i in range(n):
        mask[i, :] = torch.sum(seg_mask[i, :, label[i, :].bool()], dim=-1)
    # [N, T]
    mask = torch.clamp_max(mask, 1.0)
    # [N]
    pos_num = torch.clamp_min(torch.sum(mask, dim=1), 1.0)
    neg_num = torch.clamp_min(torch.sum(1.0 - mask, dim=1), 1.0)

    pos_loss = ((((1.0 - torch.clamp_min(aas_score, eps) ** q) / q) * mask).sum(dim=-1) / pos_num).mean()
    neg_loss = ((((1.0 - torch.clamp_min(1.0 - aas_score, eps) ** q) / q) * (1.0 - mask)).sum(dim=-1) / neg_num).mean()
    return pos_loss + neg_loss



class DecoderModel(nn.Module):
    def __init__(self, input_dim, num_classes,
                 num_layers, num_f_maps, time_emb_dim, kernel_size, dropout_rate):
        super(DecoderModel, self).__init__()

        self.time_emb_dim = time_emb_dim

        self.time_in = nn.ModuleList([
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim)
        ])

        self.conv_in = nn.Conv1d(num_classes, num_f_maps, 1)
        self.module = MixedCovAttModuleV2(num_layers, num_f_maps, input_dim, kernel_size, dropout_rate, time_emb_dim)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, t, event):
        time_emb = get_timestep_embedding(t, self.time_emb_dim)
        time_emb = self.time_in[0](time_emb)
        time_emb = swish(time_emb)
        time_emb = self.time_in[1](time_emb)

        # print("event=",event.shape)
        fra = self.conv_in(event)
        fra = self.module(fra, x, time_emb)
        event_out = self.conv_out(fra)

        return event_out


class MixedCovAttModuleV2(nn.Module):  # for decoder
    def __init__(self, num_layers, num_f_maps, input_dim_cross, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedCovAttModuleV2, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedCovAttentionLayerV2(num_f_maps, input_dim_cross, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  # 2 ** i

    def forward(self, x, x_cross, time_emb=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:, :, None]

        for layer in self.layers:
            x = layer(x, x_cross)

        return x


class MixedCovAttentionLayerV2(nn.Module):
    def __init__(self, d_model, d_cross, kernel_size, dilation, dropout_rate):
        super(MixedCovAttentionLayerV2, self).__init__()
        self.d_model = d_model
        self.d_cross = d_cross
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation

        assert (self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)
        # print("self.att_linear_q=", self.att_linear_q)
        # print("self.att_linear_k=", self.att_linear_k)
        # print("self.att_linear_v=", self.att_linear_v)
        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None

    def get_attn_indices(self, l, device):

        attn_indices = []

        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)
            # 3  3  3 ...                             (k=3, //1)
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)

            indices = [i + self.padding for i in range(s, e, step)]

            attn_indices.append(indices)

        attn_indices = np.array(attn_indices)

        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)

    def attention(self, x, x_cross):

        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)

        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2], :], (-1,))
        """
        print("x=",x.shape)
        print("x_cross=", x_cross.shape)
        print("torch.cat([x, x_cross], 1)=", torch.cat([x, x_cross], 1).shape)

        x= torch.Size([16, 128, 750])
        x_cross= torch.Size([16, 20, 750])
        torch.cat([x, x_cross], 1)= torch.Size([16, 148, 750])
        """
        x_q = self.att_linear_q(torch.cat([x, x_cross], 1))
        x_k = self.att_linear_k(torch.cat([x, x_cross], 1))
        x_v = self.att_linear_v(x)

        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)

        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1]))

        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)

        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2], :] >= self.padding,
            self.attn_indices[:x.shape[2], :] < att.shape[1] + self.padding
        )  # 1 keep, 0 mask

        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2)
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)

        return r

    def forward(self, x, x_cross):

        x_drop = self.dropout(x)
        x_cross_drop = self.dropout(x_cross)

        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop, x_cross_drop)

        out = self.ffn_block(self.norm(out1 + out2))

        return x + out