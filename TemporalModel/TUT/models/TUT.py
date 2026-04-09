import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.MyLayers import RPE, LocalAttention
from ..utils import KL_loss, class2boundary, create_distribution_from_cls
from ....template import ModelTemplate, TrainerTemplate
from ......utils import LOGITS, ATTENTION_WEIGHTS, Cholec80


class EncoderLayer(nn.Module):
    def __init__(self, l_seg, window_size, d_model, h, d_ff, activation, attention_dropout, ffn_dropout, rpe=None, pre_layernorm=False, return_attn=False):
        super(EncoderLayer, self).__init__()
        self.l_seg = l_seg
        self.window_size = window_size
        self.attention = LocalAttention(d_model, d_model, d_model, d_model, h, attention_dropout, rpe)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.InstanceNorm1d(d_model, track_running_stats=False)  # self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.InstanceNorm1d(d_model, track_running_stats=False)  # self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(ffn_dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.pre_layernorm = pre_layernorm
        self.return_attn = return_attn

    def forward(self, x, mask):
        # x: (B, d_model, L)
        # out: (B, d_model, L)

        residual = x
        if self.pre_layernorm:
            x = self.norm1(x)  # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, attn = self.attention(x, x, x, mask, self.l_seg, self.window_size, self.return_attn)
        x = residual + x
        if not self.pre_layernorm:
            x = self.norm1(x)  # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
            
        residual = x
        if self.pre_layernorm:
            x = self.norm2(x)  # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv2(self.dropout(self.activation(self.conv1(x))))
        x = residual + x
        if not self.pre_layernorm:
            x = self.norm2(x)  # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x, attn


class DecoderLayer(nn.Module):
    def __init__(self, l_seg, window_size, d_model, h, d_ff, activation, attention_dropout, ffn_dropout, rpe=None, pre_layernorm=False, return_attn=False):
        super(DecoderLayer, self).__init__()
        self.l_seg = l_seg
        self.window_size = window_size
        self.attention = LocalAttention(d_model, d_model, d_model, d_model, h, attention_dropout, rpe)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.InstanceNorm1d(d_model, track_running_stats=False)  # self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.InstanceNorm1d(d_model, track_running_stats=False)  # self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(ffn_dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.pre_layernorm = pre_layernorm
        self.return_attn = return_attn

    def forward(self, x, cross, mask):
        # x: (B, d_model, L)
        # cross: (B, d_model, L)
        # out: (B, d_model, L)

        residual = x
        if self.pre_layernorm:
            x = self.norm1(x)  # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, attn = self.attention(x, x, cross, mask, self.l_seg, self.window_size, self.return_attn)
        x = residual + x
        if not self.pre_layernorm:
            x = self.norm1(x)  # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        residual = x
        if self.pre_layernorm:
            x = self.norm2(x)  # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv2(self.dropout(self.activation(self.conv1(x))))
        x = residual + x
        if not self.pre_layernorm:
            x = self.norm2(x)  # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x, attn


class TUTSingleStage(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers, l_seg, window_size, d_model, n_heads, d_ff, activation, attention_dropout, ffn_dropout, input_dropout, rpes, pre_norm, baloss):
        super(TUTSingleStage, self).__init__()

        # self.pos = LearnableAPE(d_model)
        # self.pos = SinusoidalAPE(d_model)
        self.conv_1x1_in = nn.Conv1d(input_dim, d_model, 1)
        self.encoder_layers = nn.ModuleList([
                EncoderLayer(
                    l_seg=l_seg,
                    window_size=window_size,
                    d_model=d_model,
                    h=n_heads,
                    d_ff=d_ff,
                    activation=activation,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    rpe=rpes[i] if rpes is not None else None,
                    pre_layernorm=pre_norm,
                    return_attn=baloss,
                ) for i in range(num_layers)
            ])
        self.decoder_layers = nn.ModuleList([
                DecoderLayer(
                    l_seg=l_seg,
                    window_size=window_size,
                    d_model=d_model,
                    h=n_heads,
                    d_ff=d_ff,
                    activation=activation,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    rpe=None,
                    pre_layernorm=pre_norm,
                    return_attn=baloss,
                ) for _ in range(num_layers)
            ])
        
        self.num_layers = num_layers
        self.input_dropout = input_dropout
        self.dropout2d = nn.Dropout2d(p=input_dropout)
        self.conv_out_layer = nn.Conv1d(d_model, num_classes, 1)

    def forward(self, x, mask):
        # x: [B, D, L]
        # mask: [B, L]
        _, _, L = x.shape

        if self.input_dropout > 0:  # only for Prediction Generation Stage
            x = x.unsqueeze(2)
            x = self.dropout2d(x)
            x = x.squeeze(2)

        x = self.conv_1x1_in(x)
        # x = self.pos(x)

        len_list = [L // (2 ** i) for i in range(self.num_layers+1)]  # [L, L//2, L//4, ...]
        x_down = [x]  # the list of all the encoders' outputs, corresponding to different resolutions
        mask_down = [mask]  # the list of all the masks, corresponding to different resolutions
        attns = []  # the list of all the attention maps


        # encoder
        for i in range(self.num_layers):
            # print(x.shape)
            # down sample
            x = F.interpolate(x, size=len_list[i+1], mode='nearest')  # (B, D, L) -> (B, D, L//2)  ,  x = nn.MaxPool1d(2)(x)
            mask = F.interpolate(mask.float().unsqueeze(1), size=len_list[i + 1], mode='nearest').squeeze(1)
            x, attn = self.encoder_layers[i](x, mask)
            attns.append(attn)
            x_down.append(x)
            mask_down.append(mask)
        
        # x = self.mid_layer(x)

        # decoder
        for i in range(self.num_layers):
            # print(x.shape)
            # up sample
            x = F.interpolate(x, size=len_list[self.num_layers-i-1], mode='nearest')
            x, attn = self.decoder_layers[i](x, x_down[self.num_layers-i-1], mask_down[self.num_layers-i-1])
            attns.append(attn)

        out = self.conv_out_layer(x) * mask_down[0].unsqueeze(1)

        return out, attns


class TUT(nn.Module, ModelTemplate):
    def __init__(self,
                 input_dim,
                 num_classes,
                 l_seg=300,
                 d_model=64,
                 window_size=31,  # 25, 51
                 baloss=False,
                 d_ffn=64,
                 activation="relu",
                 pre_norm=False,
                 n_heads=4,
                 num_layers_PG=5,
                 num_R=1,
                 num_layers_R=5,
                 attention_dropout=0.2,
                 ffn_dropout=0.3,
                 input_dropout=0.4,
                 rpe_use=True,
                 rpe_share=True,
                 ):
        super(TUT, self).__init__()

        if rpe_use:
            if rpe_share:  # scale-shared rpe
                assert num_layers_PG == num_layers_R
                self.rpe_modules = nn.ModuleList([
                    RPE(window_size, h=n_heads) for _ in range(num_layers_PG)
                ])
            else: # without share
                self.rpe_modules = [nn.ModuleList([RPE(window_size, h=n_heads) for _ in range(num_layers_PG)])] + [
                    nn.ModuleList([RPE(window_size, h=n_heads) for _ in range(num_layers_R)]) for _ in range(num_R)
                ]
        else:  # without rpe
            self.rpe_modules = None

        self.PG = TUTSingleStage(
            input_dim=input_dim,
            num_classes=num_classes,
            num_layers=num_layers_PG,
            l_seg=l_seg,
            window_size=window_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ffn,
            activation=activation,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            input_dropout=input_dropout,
            rpes=self.rpe_modules if not isinstance(self.rpe_modules, list) else self.rpe_modules[0],
            pre_norm=pre_norm,
            baloss=baloss
        )

        self.Rs = nn.ModuleList([TUTSingleStage(
            input_dim=num_classes,
            num_classes=num_classes,
            num_layers=num_layers_R,
            l_seg=l_seg,
            window_size=window_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ffn,
            activation=activation,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            input_dropout=0,
            rpes=self.rpe_modules if not isinstance(self.rpe_modules, list) else self.rpe_modules[i+1],
            pre_norm=pre_norm,
            baloss=baloss
        )   for i in range(num_R)
        ])

        # self.init_emb()

        self.causal_model = False
        self.channels_last = False

    def forward(self, x, mask):
        all_attns = []
        out, attn_PG = self.PG(x, mask)
        all_attns.append(attn_PG)
        outputs = (out, )
        for R in self.Rs:
            out, attn_R = R(F.softmax(out, dim=1) * mask.unsqueeze(1), mask)  # (B, num_classes, L)
            all_attns.append(attn_R)
            outputs += (out, )

        return {
            LOGITS: outputs,
            ATTENTION_WEIGHTS: all_attns
        }

    def init_emb(self):
        """ Initialize weights of Linear layers """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


class TrainerTUT(TrainerTemplate):
    def __init__(self, phase_recognition_loss, smooth_logits_loss, phase_recognition_factor, smooth_logits_factor,
                 boundary_awareness_factor, window_size, ba_loss_in_decoder, device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        deep_supervision = True
        super().__init__(deep_supervision, phase_recognition_loss, smooth_logits_loss,
                         phase_recognition_factor, smooth_logits_factor, device_cpu, device_gpu, nplot, dataset)

        self.loss_factors["boundary_awareness"] = boundary_awareness_factor
        if boundary_awareness_factor > 0:
            self.loss_keys.append("loss_boundary_awareness")
        self.window_size = window_size
        self.ba_loss_in_decoder = ba_loss_in_decoder

    def process_batch(self, model, batch, train=True):
        input, in_valid_mask, target, nelems, N, results = model.parse_batch(batch, self.device_gpu, train)

        input = input.permute(0, 2, 1)  # N x S x C --> N x C x S
        out = model(input, in_valid_mask)
        all_logits = out[LOGITS]
        all_attns = out[ATTENTION_WEIGHTS]

        total_loss = 0
        loss_factor_sum = 0

        # standard loss
        nlogits = len(all_logits)
        for i in range(0 if self.deep_supervision else (nlogits - 1), nlogits):
            # assumes that 'refinement' of predicted logits increases with index i
            # --> logits at final index correspond to overall best predictions
            logits = all_logits[i]

            logits_chF = logits  # N x C x S
            logits_chL = logits.permute(0, 2, 1)  # N x C x S --> N x S x C; channels last format
            del logits

            loss_ = self.loss_functions["phase_recognition"](logits_chF, target) * self.loss_factors["phase_recognition"]
            total_loss = total_loss + loss_
            loss_factor_sum += self.loss_factors["phase_recognition"]

            if "loss_phase_recognition" not in results:
                results["loss_phase_recognition"] = [loss_.item(), nelems]
            else:
                results["loss_phase_recognition"][0] = results["loss_phase_recognition"][0] + loss_.item()
                results["loss_phase_recognition"][1] = results["loss_phase_recognition"][1] + nelems

            if self.loss_factors["smooth_logits"] > 0:
                loss_ = self.loss_functions["smooth_logits"](logits_chL, in_valid_mask) * self.loss_factors["smooth_logits"]
                total_loss = total_loss + loss_
                loss_factor_sum += self.loss_factors["smooth_logits"]

                if "loss_smooth" not in results:
                    results["loss_smooth"] = [loss_.item(), nelems - N]
                else:
                    results["loss_smooth"][0] = results["loss_smooth"][0] + loss_.item()
                    results["loss_smooth"][1] = results["loss_smooth"][1] + (nelems - N)

        self.get_prediction(logits_chF, in_valid_mask, target, results, nelems, train)

        if self.loss_factors["boundary_awareness"] > 0:
            # Compute boundary awareness loss
            # Code from https://github.com/ddz16/TUT/blob/main/exp/exp_TUT.py
            baloss = 0
            use_chi = False
            loss_layer_num = 1
            #    (1,L)      (begin_length)   (end_length)
            # extract from all layers (different resolution) to get begin_index and end_index
            _, begin_index, end_index = class2boundary(target)
            down_target = target
            begin_index_list = [begin_index]
            end_index_list = [end_index]
            B, _, L = input.shape
            # print(L)
            len_list = [L // (2 ** i) for i in range(loss_layer_num + 1)]  # [L, L/2, L//4, ...]
            for i in range(loss_layer_num):
                down_target = F.interpolate(down_target.float().unsqueeze(0), size=len_list[i+1]).squeeze(0).long()
                _, begin_index, end_index = class2boundary(down_target)
                begin_index_list.append(begin_index)
                end_index_list.append(end_index)

            for attn in all_attns:  # each attn is each stage list
                # attn: a list of (B, H, L, window_size)
                cnt = 0
                for i in range(loss_layer_num):
                    # print(begin_index_list[i+1])
                    if begin_index_list[i+1].shape[0] > 0 and end_index_list[i+1].shape[0] > 0:
                        attn_begin = torch.index_select(attn[i], dim=2, index=begin_index_list[i+1].to(attn[i].device))  # (B,H,l,window_size), encoder layer attn begin
                        attn_end = torch.index_select(attn[i], dim=2, index=end_index_list[i+1].to(attn[i].device))  # (B,H,l,window_size), encoder layer attn end
                        baloss = baloss + KL_loss(attn_begin, create_distribution_from_cls(0, self.window_size, use_chi).to(attn_begin.device))
                        baloss = baloss + KL_loss(attn_end, create_distribution_from_cls(2, self.window_size, use_chi).to(attn_end.device))
                        cnt += 2
                    # print(attn_begin)
                    # print(attn_end)
                    # print(baloss)

                        if self.ba_loss_in_decoder:
                            attn_begin = torch.index_select(attn[-i-1], dim=2, index=begin_index_list[i].to(attn[i].device))  # (1,H,l,window_size), decoder layer attn begin
                            attn_end = torch.index_select(attn[-i-1], dim=2, index=end_index_list[i].to(attn[i].device))  # (1,H,l,window_size), decoder layer attn begin
                            baloss = baloss + KL_loss(attn_begin, create_distribution_from_cls(0, self.window_size, use_chi).to(attn_begin.device))
                            baloss = baloss + KL_loss(attn_end, create_distribution_from_cls(2, self.window_size, use_chi).to(attn_end.device))
                            cnt += 2
                baloss = baloss / cnt

                loss_ = baloss * self.loss_factors["boundary_awareness"]
                total_loss = total_loss + loss_
                loss_factor_sum += self.loss_factors["boundary_awareness"]

                if "loss_boundary_awareness" not in results:
                    results["loss_boundary_awareness"] = [loss_.item(), cnt * self.window_size]
                else:
                    results["loss_boundary_awareness"][0] = results["loss_boundary_awareness"][0] + loss_.item()
                    results["loss_boundary_awareness"][1] = results["loss_boundary_awareness"][1] + (cnt * self.window_size)

                # break # comment on 50Salads and GTEA, meaning use all stages; if not comment, meaning only use prediction stage

        total_loss = total_loss / loss_factor_sum  # normalize loss factors

        if train is True:
            results["total_loss"] = total_loss

        return results
