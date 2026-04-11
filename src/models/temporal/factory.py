"""Factory for temporal models (TemporalModel folder is source of truth)."""

from __future__ import annotations

import importlib
from typing import Any


# Map model name → (module path, class name, input_format)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "mstcn":            ("TemporalModel.MSTCN.mstcn", "MultiStageModel", "NCT"),
    "asformer":         ("TemporalModel.ASFormer.ASFormer", "MyTransformer", "NCT"),
    "asformer_causal":  ("TemporalModel.ASFormerCausal.ASFormerCausal_clean", "ASFormerCausal", "NCT"),
    "matransformer":    ("TemporalModel.MaTransformer.model", "MaTransformer", "NCT"),
    "opera":            ("TemporalModel.Opera.opera", "OperaTransformerEncoder", "NTC"),
    "sahc":             ("TemporalModel.SAHC.hierarch_tcn2", "Hierarch_TCN2", "NTC"),
    "actionformer":     ("TemporalModel.ActionFormer.baseline", "ActionFormer", "NCT"),
    "mamba":            ("TemporalModel.MaTransformer.mamba_model", "MambaTemporalModel", "NCT"),
    "mamba_multistage": ("TemporalModel.MaTransformer.mamba_model", "MultiStageMambaModel", "NCT"),
    "trans_svnet":      ("TemporalModel.TransSVNet.trans_svnet", "TransSVNetModel", "NTC"),
    "tut":              ("TemporalModel.TUT.models.TUT", "TUT", "NCT"),
    "mstunes":          ("TemporalModel.tunes.ms_tunes", "MsTUNeS", "NTC"),
}


def get_input_format(model_name: str) -> str:
    key = model_name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown temporal model: {model_name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[key][2]


def build_temporal_model(
    model_name: str,
    feature_dim: int,
    num_classes: int,
    max_seq_len: int = 8000,
    **kwargs: Any,
):
    """
    Instantiate a temporal model by name.

    Args:
        model_name:   One of the registered model names (case-insensitive).
        feature_dim:  Input feature dimension C.
        num_classes:  Number of output phase classes.
        max_seq_len:  Maximum sequence length (used by OperA, ActionFormer).
        **kwargs:     Model-specific hyper-parameters (passed through).

    Returns:
        model: The instantiated nn.Module with `.input_format` attribute set.
    """
    key = model_name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown temporal model: {model_name!r}. Available: {list(_REGISTRY)}")

    module_path, class_name, input_format = _REGISTRY[key]
    mod = importlib.import_module(module_path)
    ModelClass = getattr(mod, class_name)

    model = _construct(key, ModelClass, feature_dim, num_classes, max_seq_len, kwargs)
    model.input_format = input_format
    return model


# ---------------------------------------------------------------------------
# Private per-model constructors
# ---------------------------------------------------------------------------

def _construct(key: str, Cls, feature_dim: int, num_classes: int, max_seq_len: int, kw: dict):
    if key == "mstcn":
        return Cls(
            num_stages=kw.get("num_stages", 2),
            num_layers=kw.get("num_layers", 9),
            num_f_maps=kw.get("num_f_maps", 64),
            dim=feature_dim,
            num_classes=num_classes,
            causal_model=kw.get("causal", True),
        )

    elif key == "asformer":
        return Cls(
            num_decoders=kw.get("num_decoders", 3),
            num_layers=kw.get("num_layers", 11),
            r1=kw.get("r1", 2),
            r2=kw.get("r2", 2),
            num_f_maps=kw.get("num_f_maps", 64),
            input_dim=feature_dim,
            num_classes=num_classes,
            channel_masking_rate=kw.get("channel_masking_rate", 0.3),
        )

    elif key == "asformer_causal":
        return Cls(
            num_decoders=kw.get("num_decoders", 3),
            num_layers=kw.get("num_layers", 11),
            r1=kw.get("r1", 2),
            r2=kw.get("r2", 2),
            num_f_maps=kw.get("num_f_maps", 64),
            input_dim=feature_dim,
            num_classes=num_classes,
            channel_masking_rate=kw.get("channel_masking_rate", 0.3),
        )

    elif key == "matransformer":
        return Cls(
            num_decoders=kw.get("num_decoders", 3),
            num_layers=kw.get("num_layers", 11),
            r1=kw.get("r1", 2),
            r2=kw.get("r2", 2),
            num_f_maps=kw.get("num_f_maps", 64),
            input_dim=feature_dim,
            num_classes=num_classes,
            channel_masking_rate=kw.get("channel_masking_rate", 0.3),
        )

    elif key == "opera":
        return Cls(
            num_class=num_classes,
            feature_dim=feature_dim,
            max_seq_len=max_seq_len,
            d_model=kw.get("d_model", 64),
            num_layers=kw.get("num_layers", 11),
            nhead=kw.get("nhead", 1),
            dim_feedforward=kw.get("dim_feedforward", 256),
            dropout=kw.get("dropout", 0.1),
            causal=kw.get("causal", True),
        )

    elif key == "sahc":
        class _SAHCArgs:
            fpn = kw.get("fpn", True)
            output = kw.get("output", True)
            feature = kw.get("feature", True)
            hier = kw.get("hier", True)
            trans = kw.get("trans", True)
            positional_encoding_type = kw.get("positional_encoding_type", "learned")
            head_num = kw.get("head_num", 8)
            embed_num = kw.get("embed_num", 512)
            block_num = kw.get("block_num", 2)

        return Cls(
            args=_SAHCArgs(),
            num_layers_PG=kw.get("num_layers_PG", 10),
            num_layers_R=kw.get("num_layers_R", 10),
            num_R=kw.get("num_R", 3),
            num_f_maps=kw.get("num_f_maps", 64),
            dim=feature_dim,
            num_classes=num_classes,
            max_len=max_seq_len,
            causal_conv=kw.get("causal", True),
        )

    elif key == "actionformer":
        return Cls(
            input_dim=feature_dim,
            max_seq_len=max_seq_len,
            num_classes=num_classes,
            d_model=kw.get("d_model", 64),
            n_head=kw.get("n_head", 4),
            n_mha_win_size=kw.get("n_mha_win_size", 9),
            scale_factor=kw.get("scale_factor", 2),
        )

    elif key == "mamba":
        return Cls(
            input_dim=feature_dim,
            num_classes=num_classes,
            d_model=kw.get("d_model", 256),
            num_layers=kw.get("num_layers", 4),
            d_state=kw.get("d_state", 16),
            d_conv=kw.get("d_conv", 4),
            expand=kw.get("expand", 2),
            dropout=kw.get("dropout", 0.1),
            bidirectional=kw.get("bidirectional", True),
        )

    elif key == "mamba_multistage":
        return Cls(
            num_stages=kw.get("num_stages", 3),
            input_dim=feature_dim,
            num_classes=num_classes,
            d_model=kw.get("d_model", 256),
            num_layers=kw.get("num_layers", 4),
            d_state=kw.get("d_state", 16),
            d_conv=kw.get("d_conv", 4),
            expand=kw.get("expand", 2),
            dropout=kw.get("dropout", 0.1),
            bidirectional=kw.get("bidirectional", True),
        )

    elif key == "trans_svnet":
        return Cls(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_stages=kw.get("num_stages", 2),
            num_layers=kw.get("num_layers", 9),
            num_f_maps=kw.get("num_f_maps", 64),
            causal_model=kw.get("causal", True),
            local_window=kw.get("local_window", 30),
            tecno_weights_path=kw.get("tecno_weights_path", None),
        )

    elif key == "tut":
        return Cls(
            input_dim=feature_dim,
            num_classes=num_classes,
            l_seg=kw.get("l_seg", 300),
            d_model=kw.get("d_model", 64),
            window_size=kw.get("window_size", 31),
            baloss=kw.get("baloss", False),
            d_ffn=kw.get("d_ffn", 64),
            activation=kw.get("activation", "relu"),
            pre_norm=kw.get("pre_norm", False),
            n_heads=kw.get("n_heads", 4),
            num_layers_PG=kw.get("num_layers_PG", 5),
            num_R=kw.get("num_R", 1),
            num_layers_R=kw.get("num_layers_R", 5),
            attention_dropout=kw.get("attention_dropout", 0.2),
            ffn_dropout=kw.get("ffn_dropout", 0.3),
            input_dropout=kw.get("input_dropout", 0.4),
            rpe_use=kw.get("rpe_use", True),
            rpe_share=kw.get("rpe_share", True),
        )

    elif key == "mstunes":
        # Align with my_opts.py defaults and structure
        conv_block_cfg = kw.get("conv_block_cfg", {
            "kernel_size": kw.get("conv_kernel", 3),
            "activation": kw.get("conv_activation", "gelu"),
            "dropout": kw.get("conv_dropout", 0.0),
            "init_method": kw.get("conv_init", "default"),
        })
        attn_cfg = kw.get("attn_cfg", {
            "dim_expansion": kw.get("attn_expansion", 4),
            "nheads": kw.get("attn_nhead", 4),
            "attn_dropout": kw.get("attn_dropout", 0.0),
            "proj_bias": kw.get("attn_proj_bias", False),
            "relative_position_bias": kw.get("attn_relative", False),
            "init_method": kw.get("transformer_init", "default"),
        })
        mlp_cfg = kw.get("mlp_cfg", {
            "dim_expansion": kw.get("ff_expansion", 1),
            "activation": kw.get("ff_activation", "gelu"),
            "dropout": kw.get("ff_dropout", 0.0),
        })
        transformer_cfg = kw.get("transformer_cfg", {
            "nlayers": kw.get("transformer_nlayers", 2),
            "normalize": kw.get("transformer_normalize", False),
            "init_method": kw.get("transformer_init", "default"),
            "residual_dropout": kw.get("transformer_residual_dropout", 0.0),
            "use_sinusoidal_pe": kw.get("sinusoidal_pe", False),
        })
        conv_attn_block_cfg = kw.get("conv_attn_block_cfg", {
            "skip_conv": kw.get("skip_conv", False),
            "skip_attn": kw.get("skip_attn", False),
        })

        return Cls(
            d_in=feature_dim,
            num_class=num_classes,
            causal_model=kw.get("causal", True),
            down_up_cfg=kw.get("down_up_cfg", {}),
            conv_block_cfg=conv_block_cfg,
            attn_cfg=attn_cfg,
            mlp_cfg=mlp_cfg,
            conv_attn_block_cfg=conv_attn_block_cfg,
            transformer_cfg=transformer_cfg,
            max_seq_len=max_seq_len,
            num_stages=kw.get("num_stages", 2),
            d_model=kw.get("d_model", 64),
            transformer_add_tokens=kw.get("transformer_add_tokens", True),
            down_blocks=kw.get("down_blocks", [2, 2, 2]),
            up_blocks=kw.get("up_blocks", [2, 2, 2]),
            temporal_scales=kw.get("temporal_scales", [3, 3, 2]),
            channel_scales=kw.get("channel_scales", [1, 1, 1]),
            up_kernels=kw.get("up_kernels", []),
            up_dilations=kw.get("up_dilations", []),
            skip_connections=kw.get("skip_connections", True),
            weighted_fusion=kw.get("weighted_fusion", True),
            fusion_weight_init=kw.get("fusion_weight_init", 1.0),
            forward_features=kw.get("forward_features", False),
        )

    raise ValueError(f"No constructor defined for {key!r}")
