import argparse
import glob
import os
import warnings
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

'''
This script is for plotting the per video result for multiple models in bar plot.

'''
Experiments = {
    "SV-RCNet": "/projects/prjs1363/SurgPhaseBench/outputs/20260413_End2End_ResNet50LSTM_split1_ramie/test_results/predictions/",
    "TMRNet": "/projects/prjs1363/SurgPhaseBench/outputs/20260414_tmrnet_resnet50_ramie_split1/test_results/predictions/",
    "TeCNO": "/projects/prjs1363/SurgPhaseBench/outputs/20260412_Stage3_MSTCN_resnet50_RAMIE_split1/test_results/predictions/",
    "Trans-SVNet": "/projects/prjs1363/SurgPhaseBench/outputs/20260414_temporal_trans_svnet_resnet50_RAMIE_split1/test_results/predictions/",
    "Causal-Transformer": "/projects/prjs1363/SurgPhaseBench/outputs/20260412_Stage3_ASFormer_Causal_ResNet50_RAMIE_split1/test_results/predictions/",
    "DINO+TeCNO": "/projects/prjs1363/SurgPhaseBench/outputs/20260412_Stage3_MSTCN_DINOv2_RAMIE_split1/test_results/predictions/",
    "DINO+Causal-Transformer": "/projects/prjs1363/SurgPhaseBench/outputs/20260412_Stage3_ASFormer_Causal_DINOv2_RAMIE_split1/test_results/predictions/",
    }

PlotsDir = "/projects/prjs1363/SurgPhaseBench/Evaluation/plots_multi_model"

# in the predictions folder there are per video txt files, we will read the gt and pred for each frame and calculate the accuracy for each video, then plot the accuracy for each video in a bar plot.

# -----------------------------
# Plotting helpers (from eval_phase.py)
# -----------------------------
def make_seaborn_palette(num_classes):
    """Maps class indices to a custom RGB palette (uint8)."""
    custom_colors = [
        "#FFD700",  # 0 Vivid Yellow
        "#E6194B",  # 1 Strong Red
        "#4363D8",  # 2 Strong Blue
        "#3CB44B",  # 3 Vivid Green
        "#F58231",  # 4 Strong Orange
        "#911EB4",  # 5 Strong Purple
        "#46F0F0",  # 6 Strong Cyan
        "#FFFAC8",  # 7 Vivid Orange
        "#FFA07A",  # 8 Light Salmon
        "#FFFF00",  # 9 Yellow
        "#00FF00",  # 10 Lime Green
        "#1D1E3E",  # 11 Very Dark Blue
        "#AAAAAA",  # 12 Gray
        "#FFFFFF",  # 13 White (separator line)
    ]

    if num_classes > len(custom_colors):
        repeats = int(np.ceil(num_classes / len(custom_colors)))
        custom_colors = (custom_colors * repeats)[:num_classes]

    try:
        import seaborn as sns
        palette = np.array(sns.color_palette(custom_colors, num_classes)) * 255
    except Exception:
        from matplotlib.colors import to_rgb
        palette = np.array([to_rgb(c) for c in custom_colors[:num_classes]]) * 255

    return palette.astype(np.uint8)


def convert_arr2img(array, row_height=300):
    """Convert 1D label sequence into a thick horizontal bar image."""
    arr = np.asarray(array, dtype=np.uint8)
    return np.tile(arr, (row_height, 1))


def _parse_key_value_int(token, expected_key):
    key, value = token.split("=", 1)
    if key != expected_key:
        raise ValueError(f"Expected token '{expected_key}=...', got '{token}'.")
    return int(value)


def parse_prediction_txt(file_path):
    """Parse one paired GT/pred txt file."""
    video_id = os.path.splitext(os.path.basename(file_path))[0]
    y_true = []
    y_pred = []

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return None, None, None

    header = lines[0].split("\t")
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 4: continue
        try:
            gt = _parse_key_value_int(parts[1], "gt")
            pred = _parse_key_value_int(parts[2], "pred")
        except: continue
        y_true.append(gt)
        y_pred.append(pred)

    return video_id, np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def plot_multi_model_prediction(video_id, y_true, model_preds, nlabels, out_png):
    """
    Save one image visualizing GT and multiple model predictions.
    model_preds: dict of {model_name: y_pred}
    """
    palette = make_seaborn_palette(nlabels + 1)
    cmap = ListedColormap((palette / 255.0), name="phase_palette")

    # Background class for white separator
    sep_height = 80
    row_height = 300

    bars = []
    labels = []
    positions = []

    # Ground Truth first
    gt_bar = convert_arr2img(y_true, row_height=row_height)
    bars.append(gt_bar)
    labels.append("Ground Truth")
    current_pos = row_height * 0.5
    positions.append(current_pos)

    # Separator
    sep_bar = np.ones((sep_height, gt_bar.shape[1]), dtype=np.uint8) * nlabels

    # Models
    current_y_offset = row_height
    for model_name, y_pred in model_preds.items():
        # Separator before next model
        bars.append(sep_bar)
        current_y_offset += sep_height

        if y_pred.shape[0] != y_true.shape[0]:
            # Simple length alignment if off by 1
            if y_pred.shape[0] == y_true.shape[0] - 1:
                y_pred = np.append(y_pred, y_pred[-1])
            elif y_pred.shape[0] > y_true.shape[0]:
                y_pred = y_pred[:y_true.shape[0]]
            else:
                print(f"Warning: model {model_name} pred length {y_pred.shape[0]} != GT {y_true.shape[0]} for {video_id}")
                continue
        
        m_bar = convert_arr2img(y_pred, row_height=row_height)
        bars.append(m_bar)
        labels.append(model_name)
        
        # Center of current model bar
        positions.append(current_y_offset + row_height * 0.5)
        current_y_offset += row_height

    vis_arr = np.concatenate(bars, axis=0)

    fig_w = max(12.0, min(28.0, y_true.shape[0] / 80.0))
    fig_h = 2.0 + (len(model_preds) + 1) * 1.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    ax.imshow(
        vis_arr,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=nlabels,
    )

    ax.set_title(f"Phase Recognition Comparison - Video {video_id}", fontsize=20)
    ax.set_xlabel("Frame Number", fontsize=14)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.tick_params(axis="x", labelsize=15)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Left labels
    img_width = vis_arr.shape[1]
    text_x = -0.02 * img_width # Near the edge but sensitive to figure width
    for label, pos in zip(labels, positions):
        ax.text(text_x, pos, label, fontsize=16, color="black", ha="right", va="center")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    # 1. Load data for each experiment
    video_data = {}  # {video_id: {gt: arr, preds: {model_name: pred_arr}}}
    all_videos = set()
    
    for model_name, results_dir in Experiments.items():
        print(f"Loading results for {model_name} from {results_dir}...")
        txt_files = glob.glob(os.path.join(results_dir, "*.txt"))
        for txt_file in txt_files:
            vid, y_true, y_pred = parse_prediction_txt(txt_file)
            if vid is None: continue
            
            all_videos.add(vid)
            if vid not in video_data:
                video_data[vid] = {"gt": y_true, "preds": {}}
            video_data[vid]["preds"][model_name] = y_pred

    if not all_videos:
        print("No videos found to plot.")
        return

    # 2. Infer number of labels (max found in any GT or pred)
    max_label = 0
    for vid in video_data:
        max_label = max(max_label, np.max(video_data[vid]["gt"]))
        for pred in video_data[vid]["preds"].values():
            max_label = max(max_label, np.max(pred))
    nlabels = int(max_label) + 1
    print(f"Inferred {nlabels} phase classes.")

    # 3. Plot each video
    os.makedirs(PlotsDir, exist_ok=True)
    for vid in sorted(all_videos):
        # Only plot if we have GT (from at least one model's paired files)
        if "gt" not in video_data[vid]: continue
        
        print(f"Plotting {vid}...")
        out_png = os.path.join(PlotsDir, f"{vid}_comparison.png")
        plot_multi_model_prediction(
            vid, 
            video_data[vid]["gt"], 
            video_data[vid]["preds"], 
            nlabels, 
            out_png
        )
    
    print(f"Done! Plots saved to {PlotsDir}")


if __name__ == "__main__":
    main()
