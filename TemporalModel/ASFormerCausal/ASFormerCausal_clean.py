import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
import math


def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p * idx_decoder)


class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape

        assert c1 == c2

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6)  # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention)
        attention = attention * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage):  # r1 = r2
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)

        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        assert self.stage in ['encoder', 'decoder']

        self.att_helper = AttentionHelper()
        window_mask_ = self.construct_window_mask()
        self.register_buffer('window_mask', window_mask_, persistent=False)

    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, 2l), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl * 2))
        for i in range(self.bl):
            # Causal sliding window: can only attend to previous tokens and current token (up to self.bl window)
            window_mask[:, i, i:i + self.bl] = 1
        return window_mask

    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder

        query = self.query_conv(x1)
        key = self.key_conv(x1)

        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        return self._causal_sliding_window_self_att(query, key, value, mask)

    def _causal_sliding_window_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        device = q.device
        _, c2, _ = k.size()
        v_dim, _ = v.size(1), v.size(2)

        # padding zeros for the last segment
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl), device=device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl), device=device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, v_dim, self.bl - L % self.bl), device=device)], dim=-1)
            nb += 1
        
        # Mask calculation for segments
        padding_mask = torch.cat([
            torch.ones((m_batchsize, 1, L), device=device) * mask[:, 0:1, :],
            torch.zeros((m_batchsize, 1, self.bl * nb - L), device=device)
        ], dim=-1)

        # Reshape query into segments
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)

        # Left padding for key/value/mask for causal access to previous window
        k_padded = torch.cat([torch.zeros(m_batchsize, c2, self.bl, device=device), k], dim=-1)
        v_padded = torch.cat([torch.zeros(m_batchsize, v.size(1), self.bl, device=device), v], dim=-1)
        pm_padded = torch.cat([torch.zeros(m_batchsize, 1, self.bl, device=device), padding_mask], dim=-1)

        # Extract context windows (current block + preceding block)
        k_windows = torch.cat([k_padded[:, :, i * self.bl : i * self.bl + self.bl * 2] for i in range(nb)], dim=0)
        v_windows = torch.cat([v_padded[:, :, i * self.bl : i * self.bl + self.bl * 2] for i in range(nb)], dim=0)
        pm_windows = torch.cat([pm_padded[:, :, i * self.bl : i * self.bl + self.bl * 2] for i in range(nb)], dim=0)

        # Final attention mask: sliding window mask * padding mask
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * pm_windows

        # Causal mask within the 2*bl window: 
        # For query at index t in block, we can only see k up to index (t + bl) in the concatenated window
        causal_mask = torch.ones((1, self.bl, self.bl * 2), device=device)
        for i in range(self.bl):
            causal_mask[:, i, i + self.bl + 1:] = 0
            
        final_mask = final_mask * causal_mask

        output, attention = self.att_helper.scalar_dot_att(q, k_windows, v_windows, final_mask)
        output = self.conv_out(F.relu(output))

        # Reshape back to original sequence
        # output is (nb * m_batchsize, v_out_dim, bl)
        v_out_dim = output.size(1)
        output = output.reshape(nb, m_batchsize, v_out_dim, self.bl).permute(1, 2, 0, 3).reshape(m_batchsize, v_out_dim, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class CausalConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(CausalConvFeedForward, self).__init__()
        # Kernel size 3 causal conv requires 2 * dilation padding on the left
        self.padding = (dilation * 2, 0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.pad(x, self.padding)
        x = self.conv(x)
        x = self.relu(x)
        return x


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = CausalConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, stage=stage)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, 'encoder', alpha) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, alpha):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, 'decoder', alpha) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature


class ASFormerCausal(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        super(ASFormerCausal, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, alpha=1)
        self.decoders = nn.ModuleList([
            Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, alpha=exponential_descrease(s)) 
            for s in range(num_decoders)
        ])
        
    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = [out]
        
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            outputs.append(out)
 
        return outputs
