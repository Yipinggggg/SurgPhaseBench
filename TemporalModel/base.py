import torch
from torch import nn
import torch.nn.functional as F

from .Baselines.mstcn import MultiStageModel
from .Baselines.transformer2_3_1 import Transformer as TransSVNetBlock
from .Baselines.SAHC.hierarch_tcn2 import Hierarch_TCN2
from .Baselines.SAHC.utils import fusion as upsample
from .Baselines.opera import OperaTransformerEncoder, attention_regularization_loss
from .Baselines.ASFormer import MyTransformer as ASFormerModel
from .template import ModelTemplate, TrainerTemplate
from ...utils import Cholec80, LOGITS, FEATURE_LOGITS, ATTENTION_WEIGHTS


class BaselineTeCNO(MultiStageModel, ModelTemplate):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, causal_model):
        super().__init__(num_stages, num_layers, num_f_maps, dim, num_classes, causal_model)

        self.channels_last = False


class TrainerTeCNO(TrainerTemplate):
    def __init__(self, phase_recognition_loss, smooth_logits_loss, phase_recognition_factor, smooth_logits_factor,
                 device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        deep_supervision = True
        super().__init__(deep_supervision, phase_recognition_loss, smooth_logits_loss,
                         phase_recognition_factor, smooth_logits_factor, device_cpu, device_gpu, nplot, dataset)


class TransSVNet(nn.Module, ModelTemplate):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes,
                 tecno_weights_file_path=None, local_window=30):
        super().__init__()

        self.causal_model = True
        self.channels_last = True
        self.tecno = MultiStageModel(num_stages, num_layers, num_f_maps, dim, num_classes, self.causal_model)
        self.attention_mechanism = TransSVNetBlock(mstcn_f_maps=num_f_maps, mstcn_f_dim=dim, out_features=num_classes,
                                                   len_q=local_window)

        if tecno_weights_file_path is not None:
            print('Initialize TeCNO weights...')
            tecno_weights = torch.load(tecno_weights_file_path)['model_weights']
            try:
                self.tecno.load_state_dict(tecno_weights, strict=True)
            except RuntimeError as e:
                print("An error occurred for 'strict=True'!")
                print(e)
                print("Retrying with 'strict=False'...")
                self.tecno.load_state_dict(tecno_weights, strict=False)
            print('Done.')
            self.frozen_tecno = True
        else:
            print('TeCNO weights were not initialized at model creation.')
            self.frozen_tecno = False

        if self.frozen_tecno is True:
            # freeze TeCNO weights
            for param in self.tecno.parameters():
                param.requires_grad = False
            self.tecno.eval()

    def train(self, mode: bool = True):
        if self.frozen_tecno is not True:
            super().train(mode)
        else:
            if mode is True:
                self.tecno.eval()  # keep TeCNO module with frozen weights in eval mode
                self.attention_mechanism.train()
            else:
                super().train(mode)

    def forward(self, x, mask):  # x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        # x: CNN feature sequence
        tecno_in = x.permute(0, 2, 1)  # N x C x S
        tecno_out = self.tecno(tecno_in, mask)

        out_features = tecno_out[LOGITS][-1]  # output of final tecno stage
        if self.frozen_tecno is True:
            out_features = out_features.detach()
        logits = self.attention_mechanism(out_features, x)  # no handling of padded sequences or batch size > 1
        # logits shape: S x 1 x C
        logits = logits.permute(1, 0, 2)  # N x S x C
        logits = logits * mask.unsqueeze(-1)

        if self.frozen_tecno is True:
            return {
                LOGITS: (logits, )
            }
        else:
            out = list()
            for tecno_logits in tecno_out[LOGITS]:
                out.append(tecno_logits.permute(0, 2, 1))   # N x C x S -->  N x S x C
            out.append(logits)
            return {
                LOGITS: tuple(out)
            }


class TrainerTransSVNet(TrainerTemplate):
    def __init__(self, deep_supervision, phase_recognition_loss, smooth_logits_loss,
                 phase_recognition_factor, smooth_logits_factor, device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        super().__init__(deep_supervision, phase_recognition_loss, smooth_logits_loss,
                         phase_recognition_factor, smooth_logits_factor, device_cpu, device_gpu, nplot, dataset)


class SAHC(nn.Module, ModelTemplate):
    def __init__(self, num_f_maps, dim, num_classes, max_len, skip_attn=False, causal_conv=True,
                 downsampling_mode="orig", downsampling_kernel=7, causal_upsampling=False, causal_attention=False,
                 acausal_attention_masking="none"):
        super().__init__()
        self.channels_last = True

        # SAHC default configuration
        class args: pass
        args.fpn = True
        args.output = False
        args.feature = False
        args.trans = not (skip_attn is True)
        args.hier = True
        args.positional_encoding_type = "learned"
        args.head_num = 8
        args.embed_num = 512
        args.block_num = 1
        args.last = False
        args.first = True
        num_layers_PG = 11
        num_layers_R = 10
        num_R = 3
        self.sahc = Hierarch_TCN2(
            args, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, max_len, causal_conv,
            downsampling_mode, downsampling_kernel, causal_upsampling, causal_attention, acausal_attention_masking
        )

        self.prediction_level = 0  # which predictions to use after all

    def forward(self, x, unused):  # x is of shape N x S x C
        predicted_list, feature_list, prototype = self.sahc(x)
        # predicted_list --> logits[0] (full resolution) ... logits[3] (lowest resolution)
        return {
            LOGITS: predicted_list  # all tensors of shape N x C x S
        }

    def infer_batch(self, batch, device_gpu, return_logits=False):
        input, in_valid_mask, target, _, _, results = self.parse_batch(batch, device_gpu, train=False, get_target=True)
        target = target.squeeze(0)  # squeeze batch dimension --> shape: S

        out = self.forward(input, None)
        _, resize_list, labels_list = upsample(out[LOGITS], target, None)

        all_out = resize_list[self.prediction_level]  # shape: N x C x S --> chF
        if return_logits:
            results["logits_gpu"] = all_out.permute(0, 2, 1).detach()
        else:
            _, predicted = torch.max(torch.nn.Softmax(dim=1)(all_out), dim=1)  # N x S
            results["predicted_gpu"] = predicted.detach()

        return results


class TrainerSAHC(TrainerTemplate):
    def __init__(self, device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        deep_supervision = True
        super().__init__(deep_supervision, None, None, 1, 1, device_cpu, device_gpu, nplot, dataset)

        # original implementation of loss functions from SAHC repo
        self.num_classes = Cholec80.num_phases  # TODO
        self.loss_layer = nn.CrossEntropyLoss()
        self.mse_layer = nn.MSELoss(reduction='none')

    def process_batch(self, model, batch, train=True):
        input, in_valid_mask, target, nelems, N, results = model.parse_batch(batch, self.device_gpu, train)
        target = target.squeeze(0)  # squeeze batch dimension --> shape: S

        out = model(input, None)
        _, resize_list, labels_list = upsample(out[LOGITS], target, None)

        total_loss = 0
        for p, l in zip(resize_list, labels_list):
            # phase recognition loss, factor == 1
            loss_ = self.loss_layer(p.transpose(2, 1).contiguous().view(-1, self.num_classes), l.view(-1))
            total_loss = total_loss + loss_

            if "loss_phase_recognition" not in results:
                results["loss_phase_recognition"] = [loss_.item(), nelems]
            else:
                results["loss_phase_recognition"][0] = results["loss_phase_recognition"][0] + loss_.item()
                results["loss_phase_recognition"][1] = results["loss_phase_recognition"][1] + nelems

            # smooth logits loss, factor == 1
            loss_ = torch.mean(torch.clamp(
                self.mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                min=0, max=16
            ))
            total_loss = total_loss + loss_

            if "loss_smooth" not in results:
                results["loss_smooth"] = [loss_.item(), nelems - N]
            else:
                results["loss_smooth"][0] = results["loss_smooth"][0] + loss_.item()
                results["loss_smooth"][1] = results["loss_smooth"][1] + (nelems - N)

        # no normalization of loss factors in this case

        if train is True:
            results["total_loss"] = total_loss

        all_out = resize_list[model.prediction_level]  # shape: N x C x S --> chF
        self.get_prediction(all_out, in_valid_mask, target, results, nelems, train)

        return results


class OperA(nn.Module, ModelTemplate):
    def __init__(self, num_classes, feature_dim, max_len, d_model=64, num_layers=11,
                 nhead=1, dim_feedforward=256, dropout=0.0, activation="relu", causal=True, full_attention=False):
        super().__init__()

        self.opera = OperaTransformerEncoder(num_classes, feature_dim, max_len, d_model, num_layers,
                                             nhead, dim_feedforward, dropout, activation, causal, full_attention)
        self.causal = causal
        self.full_attention = full_attention
        self.channels_last = True

    def forward(self, x, unused):
        return self.opera(x)


class TrainerOperA(TrainerTemplate):
    def __init__(self, phase_recognition_loss, smooth_logits_loss,
                 phase_recognition_factor, smooth_logits_factor, feature_logits_factor, attn_reg_factor,
                 device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        assert (phase_recognition_factor == 1)
        deep_supervision = False
        super().__init__(deep_supervision, phase_recognition_loss, smooth_logits_loss,
                         phase_recognition_factor, smooth_logits_factor, device_cpu, device_gpu, nplot, dataset)

        self.loss_factors["feature_logits"] = feature_logits_factor
        self.loss_factors["attention_regularization"] = attn_reg_factor

        self.loss_keys.append("loss_feature_classifier")
        self.loss_keys.append("loss_attention_regularization")

    def process_batch(self, model, batch, train=True):
        input, in_valid_mask, target, nelems, N, results = model.parse_batch(batch, self.device_gpu, train)

        out = model(input, None)

        total_loss = 0
        loss_factor_sum = 0

        logits = out[LOGITS][-1]
        logits_chF = logits.permute(0, 2, 1)  # N x S x C --> N x C x S; channels first format
        logits_chL = logits
        del logits

        loss_ = self.loss_functions["phase_recognition"](logits_chF, target) * self.loss_factors["phase_recognition"]
        total_loss = total_loss + loss_
        loss_factor_sum += self.loss_factors["phase_recognition"]
        results["loss_phase_recognition"] = (loss_.item(), nelems)

        if self.loss_factors["smooth_logits"] > 0:
            loss_ = self.loss_functions["smooth_logits"](logits_chL, in_valid_mask) * self.loss_factors["smooth_logits"]
            total_loss = total_loss + loss_
            loss_factor_sum += self.loss_factors["smooth_logits"]
            results["loss_smooth"] = (loss_.item(), nelems - N)

        if self.loss_factors["attention_regularization"] > 0:
            # train classifier on top of features (meaningful feature logits are required for attention regularization loss)
            feature_logits_chF = out[FEATURE_LOGITS].permute(0, 2, 1)
            loss_ = self.loss_functions["phase_recognition"](feature_logits_chF, target) * self.loss_factors["feature_logits"]
            total_loss = total_loss + loss_
            loss_factor_sum += self.loss_factors["feature_logits"]
            results["loss_feature_classifier"] = (loss_.item(), nelems)

            if model.causal is True:
                attn_weights = out[ATTENTION_WEIGHTS][0]  # attention weights in first layer
                attention_regularization_loss(
                    attn_weights, feature_logits_chF.detach(), target, attn_mask=torch.logical_not(model.opera.causal_mask)
                ) * self.loss_factors["attention_regularization"]
            else:
                if model.full_attention is True:
                    attn_weights = out[ATTENTION_WEIGHTS][0]
                    attention_regularization_loss(
                        attn_weights, feature_logits_chF.detach(), target, attn_mask=None
                    ) * self.loss_factors["attention_regularization"]
                else:
                    # regularize attention weights in first (attending to past) and second (attending to future) layer
                    loss_1 = attention_regularization_loss(
                        out[ATTENTION_WEIGHTS][0], feature_logits_chF.detach(), target, attn_mask=torch.logical_not(model.opera.future_mask)
                    ) * self.loss_factors["attention_regularization"]
                    loss_2 = attention_regularization_loss(
                        out[ATTENTION_WEIGHTS][1], feature_logits_chF.detach(), target, attn_mask=torch.logical_not(model.opera.past_mask)
                    ) * self.loss_factors["attention_regularization"]
                    loss_ = (loss_1 + loss_2) / 2
            total_loss = total_loss + loss_
            loss_factor_sum += self.loss_factors["attention_regularization"]
            results["loss_attention_regularization"] = (loss_.item(), 1)

        total_loss = total_loss / loss_factor_sum  # normalize loss factors

        if train is True:
            results["total_loss"] = total_loss

        self.get_prediction(logits_chF, in_valid_mask, target, results, nelems, train)

        return results


class ASFormer(nn.Module, ModelTemplate):

    def __init__(self, num_decoders, num_layers, num_f_maps, dim, num_classes):
        super().__init__()

        self.asformer = ASFormerModel(num_decoders=num_decoders, num_layers=num_layers, r1=2, r2=2, num_f_maps=num_f_maps,
                                      input_dim=dim, num_classes=num_classes, channel_masking_rate=0.3)
        self.causal = False
        self.channels_last = False

    def forward(self, x, mask):  # x is of shape N x C x S, mask is N x S (1 --> keep, 0 --> mask)
        outputs = self.asformer(x, mask.unsqueeze(1))

        logits = []
        for out in outputs:
            logits.append(out)

        return {
            LOGITS: tuple(logits)  # all tensors of shape N x C x S
        }


class TrainerASFormer(TrainerTemplate):
    def __init__(self, phase_recognition_loss, smooth_logits_loss, phase_recognition_factor, smooth_logits_factor,
                 device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        deep_supervision = True
        super().__init__(deep_supervision, phase_recognition_loss, smooth_logits_loss,
                         phase_recognition_factor, smooth_logits_factor, device_cpu, device_gpu, nplot, dataset)


class BaselineRNN(nn.Module, ModelTemplate):
    def __init__(self, num_layers, d_model, feature_dim, num_classes, causal_model, init_orthogonal, use_gru=True):
        super().__init__()
        self.channels_last = True

        self.use_gru = use_gru
        if use_gru is True:
            self.rnn = nn.GRU(
                input_size=feature_dim, hidden_size=d_model, num_layers=num_layers,
                batch_first=True, bidirectional=causal_model is False  # expected tensor shapes: N x S x C (channels last)
            )
        else:
            self.rnn = nn.LSTM(
                input_size=feature_dim, hidden_size=d_model, num_layers=num_layers,
                batch_first=True, bidirectional=causal_model is False  # expected tensor shapes: N x S x C (channels last)
            )
        self.classifier = nn.Linear(d_model * (2 if causal_model is False else 1), num_classes)

        self.init_orthogonal = init_orthogonal
        self._init_weights(d_model)

    def _init_weights(self, hidden_size):
        if self.init_orthogonal is True:
            if self.use_gru is True:
                for suffix in (["", "_reverse"] if self.rnn.bidirectional else [""]):
                    for layer in range(self.rnn.num_layers):
                        weight_ih = getattr(self.rnn, "weight_ih_l{}".format(layer) + suffix)
                        weight_hh = getattr(self.rnn, "weight_hh_l{}".format(layer) + suffix)
                        bias_ih = getattr(self.rnn, "bias_ih_l{}".format(layer) + suffix)
                        bias_hh = getattr(self.rnn, "bias_hh_l{}".format(layer) + suffix)

                        nn.init.orthogonal_(weight_ih)
                        nn.init.orthogonal_(weight_hh)
                        nn.init.zeros_(bias_ih)
                        nn.init.zeros_(bias_hh)
            else:
                for suffix in (["", "_reverse"] if self.rnn.bidirectional else [""]):
                    for layer in range(self.rnn.num_layers):
                        weight_ih = getattr(self.rnn, "weight_ih_l{}".format(layer) + suffix)
                        weight_hh = getattr(self.rnn, "weight_hh_l{}".format(layer) + suffix)
                        bias_ih = getattr(self.rnn, "bias_ih_l{}".format(layer) + suffix)
                        bias_hh = getattr(self.rnn, "bias_hh_l{}".format(layer) + suffix)

                        nn.init.orthogonal_(weight_ih)
                        nn.init.orthogonal_(weight_hh)
                        nn.init.zeros_(bias_ih)
                        nn.init.zeros_(bias_hh)
                        nn.init.ones_(bias_ih[hidden_size:hidden_size * 2])  # forget gates
                        nn.init.ones_(bias_hh[hidden_size:hidden_size * 2])

    def extra_repr(self) -> str:
        return 'init_orthogonal={}'.format(self.init_orthogonal)

    def forward(self, x, mask):  # x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, hidden = self.rnn(x * mask.unsqueeze(-1))
        x = self.classifier(x)
        x = x * mask.unsqueeze(-1)

        return {
            LOGITS: (x, )
        }


class TrainerRNN(TrainerTemplate):
    def __init__(self, phase_recognition_loss, smooth_logits_loss, phase_recognition_factor, smooth_logits_factor,
                 device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        deep_supervision = False
        super().__init__(deep_supervision, phase_recognition_loss, smooth_logits_loss,
                         phase_recognition_factor, smooth_logits_factor, device_cpu, device_gpu, nplot, dataset)

