# Implementation of OperA, as described in https://arxiv.org/abs/2103.03873

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGITS = "logits"
FEATURE_LOGITS = "feature_logits"
ATTENTION_WEIGHTS = "attention_weights"


def create_causal_mask(length: int) -> torch.Tensor:
    """Return a lower-triangular causal attention mask (True means attendable)."""
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


class OperaTransformerEncoder(nn.Module):
    def __init__(self, num_class, feature_dim, max_seq_len, d_model, num_layers, nhead,
                 dim_feedforward=2048, dropout=0.0, activation="relu", causal=True, full_attention=False):
        super().__init__()

        self.in_proj = nn.Linear(feature_dim, d_model)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(OperaTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation))
        self.classifier = nn.Linear(d_model, num_class)
        self.feat_classifier = nn.Linear(feature_dim, num_class)

        self.causal = causal
        self.full_attention = full_attention
        if self.causal is True:
            causal_mask = create_causal_mask(max_seq_len)  # True --> allowed to attend
            causal_mask = torch.logical_not(causal_mask)  # True --> NOT allowed to attend (PyTorch convention)
            self.register_buffer('causal_mask', causal_mask, persistent=False)
        else:
            if self.full_attention is False:
                future_mask = create_causal_mask(max_seq_len)  # allowed to attend to past positions only
                self.register_buffer('future_mask', torch.logical_not(future_mask), persistent=False)
                past_mask = torch.t(future_mask)  # allowed to attend to future positions only
                self.register_buffer('past_mask', torch.logical_not(past_mask), persistent=False)

    def extra_repr(self) -> str:
        if self.causal is False:
            return 'causal={}, separate_past_future={}'.format(self.causal, (not self.full_attention))
        else:
            return 'causal={}'.format(self.causal)

    def forward(self, x):
        fe_logits = self.feat_classifier(x)

        x = self.in_proj(x)

        attention_weights = []
        _, S, _ = x.shape
        if self.causal is True:
            attn_mask = self.causal_mask[:S, :S]
            for layer in self.layers:
                x, attn = layer(x, src_mask=attn_mask)
                attention_weights.append(attn)
        else:
            if self.full_attention is True:
                for layer in self.layers:
                    x, attn = layer(x, src_mask=None)
                    attention_weights.append(attn)
            else:
                # process past and future information separately
                attn_mask_future = self.future_mask[:S, :S]
                attn_mask_past = self.past_mask[:S, :S]
                for i, layer in enumerate(self.layers):
                    if (i % 2) == 0:
                        x, attn = layer(x, src_mask=attn_mask_future)
                    else:
                        x, attn = layer(x, src_mask=attn_mask_past)
                    attention_weights.append(attn)

        logits = self.classifier(x)

        return {
            LOGITS: (logits, ),
            FEATURE_LOGITS: fe_logits,
            ATTENTION_WEIGHTS: attention_weights
        }


class OperaTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True, norm_first=False)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        src2, weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
            need_weights=True, average_attn_weights=True
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # In the paper, it is unclear how exactly this part was implemented (simply depicted as 'FC' in Fig. 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, weights


# pay higher attention to features with lower cross entropy
def attention_regularization_loss(attention_weights, feat_logits, target, attn_mask=None):
    # expected to be attention_weights of *first* layer
    attention_weights = attention_weights.squeeze(0)  # S x S
    assert (attention_weights.ndim == 2)
    S, S = attention_weights.shape
    feat_logits = feat_logits  # 1 x C x S
    target = target  # 1 x S

    ce = F.cross_entropy(feat_logits, target, reduction="none").squeeze(0)
    assert (ce.ndim == 1 and ce.shape[0] == S)

    attn_vector = torch.sum(attention_weights, dim=0)
    if attn_mask is not None:
        normalizer = attn_mask[:S, :S].sum(dim=0)  # [S, S-1, ..., 1] or [1, ..., S-1, S]
    else:
        normalizer = S
    attn_vector = attn_vector / normalizer

    return torch.dot(attn_vector, ce)  # normalize by sequence length ?
