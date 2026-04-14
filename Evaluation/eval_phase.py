import argparse
import glob
import os
import warnings

import numpy as np
import yaml
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)


eps = 0.000001


# Optional Octave support for relaxed metrics parity checks.
OCTAVE_CHECK = False
if OCTAVE_CHECK:
    OCTAVE_PATH = "PhaseMetrics/matlab-eval"
    OCTAVE_FOUND = True
    try:
        from oct2py import octave

        octave.eval('disp("Octave found.")')
        octave.addpath(OCTAVE_PATH)
    except Exception:
        print("Could not import oct2py.octave. MATLAB parity checks are disabled.")
        OCTAVE_FOUND = False
else:
    OCTAVE_FOUND = False


# -----------------------------
# Plotting helpers
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
        # Fallback if seaborn is unavailable.
        from matplotlib.colors import to_rgb

        palette = np.array([to_rgb(c) for c in custom_colors[:num_classes]]) * 255

    return palette.astype(np.uint8)


def convert_arr2img(array, row_height=300):
    """Convert 1D label sequence into a thick horizontal bar image."""
    arr = np.asarray(array, dtype=np.uint8)
    return np.tile(arr, (row_height, 1))


def plot_single_prediction(video_id, y_true, y_pred, nlabels, out_png):
    """Save one image visualizing GT and predictions in legacy phase-bar style."""
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import MultipleLocator

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Cannot plot video {video_id}: y_true length {y_true.shape[0]} != y_pred length {y_pred.shape[0]}"
        )

    # +1 class for the white separator line.
    palette = make_seaborn_palette(nlabels + 1)
    cmap = ListedColormap((palette / 255.0), name="phase_palette")

    gt_arr = convert_arr2img(y_true, row_height=300)
    pred_arr = convert_arr2img(y_pred, row_height=300)

    sep_height = 100
    sep_arr = np.ones((sep_height, gt_arr.shape[1]), dtype=np.uint8) * nlabels
    vis_arr = np.concatenate([gt_arr, sep_arr, pred_arr], axis=0)

    fig_w = max(12.0, min(28.0, y_true.shape[0] / 80.0))
    fig_h = 8.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(
        vis_arr,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=nlabels,
    )

    ax.set_title(f"RAMIE - Patient {video_id}", fontsize=20)
    ax.set_xlabel("Frame Number", fontsize=14)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.tick_params(axis="x", labelsize=15)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Left labels (match legacy placement style).
    img_width = vis_arr.shape[1]
    text_x = -0.1 * img_width
    text_y_gt = gt_arr.shape[0] * 0.5
    text_y_pred = gt_arr.shape[0] + sep_height + pred_arr.shape[0] * 0.5

    ax.text(text_x, text_y_gt, "Ground Truth", fontsize=20, color="black", ha="right", va="center")
    ax.text(text_x, text_y_pred, "Prediction", fontsize=20, color="black", ha="right", va="center")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_predictions_from_txt_dir(predictions_dir, plots_dir, nlabels=None):
    """Generate one PNG per TXT prediction file."""
    videos = load_all_video_predictions(predictions_dir)
    if nlabels is None:
        nlabels = infer_nlabels(videos)

    os.makedirs(plots_dir, exist_ok=True)
    saved_pngs = []

    for video_id in sorted(videos.keys()):
        y_true = videos[video_id]["y_true"]
        y_pred = videos[video_id]["y_pred"]

        if y_pred.shape[0] == (y_true.shape[0] - 1):
            y_true = y_true[:-1]

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Cannot plot predictions for video {video_id}: y_true length {y_true.shape[0]} "
                f"!= y_pred length {y_pred.shape[0]}."
            )

        out_png = os.path.join(plots_dir, f"{video_id}.png")
        plot_single_prediction(video_id, y_true, y_pred, nlabels, out_png)
        saved_pngs.append(out_png)

    return saved_pngs


# -----------------------------
# Parsing helpers
# -----------------------------
def _parse_key_value_int(token, expected_key):
    key, value = token.split("=", 1)
    if key != expected_key:
        raise ValueError(f"Expected token '{expected_key}=...', got '{token}'.")
    return int(value)


def parse_prediction_txt(file_path):
    """Parse one paired GT/pred txt file.

    Expected format:
        frame\tgt\tpred\tconf
        <frame_id>\tgt=<int>\tpred=<int>\tconf=<float>
    """
    video_id = os.path.splitext(os.path.basename(file_path))[0]
    y_true = []
    y_pred = []

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty file: {file_path}")

    header = lines[0].split("\t")
    expected_header = ["frame", "gt", "pred", "conf"]
    if header != expected_header:
        raise ValueError(
            f"Unexpected header in {file_path}. "
            f"Expected {expected_header}, got {header}."
        )

    for i, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError(f"Malformed line {i} in {file_path}: '{line}'")

        try:
            gt = _parse_key_value_int(parts[1], "gt")
            pred = _parse_key_value_int(parts[2], "pred")
        except Exception as exc:
            raise ValueError(f"Could not parse line {i} in {file_path}: '{line}'") from exc

        y_true.append(gt)
        y_pred.append(pred)

    return video_id, np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def load_all_video_predictions(predictions_dir):
    txt_files = sorted(glob.glob(os.path.join(predictions_dir, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {predictions_dir}")

    videos = {}
    for txt_file in txt_files:
        video_id, y_true, y_pred = parse_prediction_txt(txt_file)
        if video_id in videos:
            raise ValueError(f"Duplicate video id '{video_id}' from file {txt_file}")
        videos[video_id] = {"y_true": y_true, "y_pred": y_pred}

    return videos


# -----------------------------
# Metric helpers (kept compatible with old report format)
# -----------------------------
def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])

    phase_labels, _ = get_phase_segments(Yi)
    assert phase_labels == Yi_split.tolist()

    return Yi_split


def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]

    _, phase_borders = get_phase_segments(Yi)
    assert [intervals[i][0] for i in range(len(intervals))] == phase_borders[:-1]
    assert [intervals[i][1] for i in range(len(intervals))] == phase_borders[1:]

    return intervals


def get_phase_segments(y_true):
    if isinstance(y_true, list):
        y_true = np.asarray(y_true)
    assert y_true.ndim == 1

    phase_labels = []
    phase_borders = []
    current_phase = -1

    for i in range(len(y_true)):
        phase = y_true[i]
        if phase != current_phase:
            phase_labels.append(phase)
            phase_borders.append(i)
            current_phase = phase

    phase_borders.append(i + 1)
    return phase_labels, phase_borders


def overlap_(p, y, n_classes, bg_class, overlap):
    true_intervals = np.array(segment_intervals(y))
    true_labels = segment_labels(y)
    pred_intervals = np.array(segment_intervals(p))
    pred_labels = segment_labels(p)

    if bg_class is not None:
        true_intervals = true_intervals[true_labels != bg_class]
        true_labels = true_labels[true_labels != bg_class]
        pred_intervals = pred_intervals[pred_labels != bg_class]
        pred_labels = pred_labels[pred_labels != bg_class]

    n_true = true_labels.shape[0]
    n_pred = pred_labels.shape[0]

    TP = np.zeros(n_classes, float)
    FP = np.zeros(n_classes, float)
    true_used = np.zeros(n_true, float)

    for j in range(n_pred):
        intersection = np.minimum(pred_intervals[j, 1], true_intervals[:, 1]) - np.maximum(
            pred_intervals[j, 0], true_intervals[:, 0]
        )
        union = np.maximum(pred_intervals[j, 1], true_intervals[:, 1]) - np.minimum(
            pred_intervals[j, 0], true_intervals[:, 0]
        )
        iou = (intersection / union) * (pred_labels[j] == true_labels)

        idx = iou.argmax()

        if iou[idx] >= overlap and not true_used[idx]:
            TP[pred_labels[j]] += 1
            true_used[idx] = 1
        else:
            FP[pred_labels[j]] += 1

    TP = TP.sum()
    FP = FP.sum()
    FN = n_true - true_used.sum()
    return TP, FP, FN


def calc_overlap_f1(y_true, y_pred, nlabels, overlap=0.5):
    assert 0 < overlap <= 1
    TP, FP, FN = overlap_(y_pred, y_true, nlabels, bg_class=None, overlap=overlap)

    if (2 * TP + FP + FN) > 0:
        return 2 * TP / (2 * TP + FP + FN)
    return 0


def levenstein_(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)

    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(
                    D[i - 1, j] + 1,
                    D[i, j - 1] + 1,
                    D[i - 1, j - 1] + 1,
                )

    if norm:
        return (1 - D[-1, -1] / max(m_row, n_col)) * 100
    return D[-1, -1]


def calc_edit_score(y_true, y_pred):
    segments_true = segment_labels(y_true).tolist()
    segments_pred = segment_labels(y_pred).tolist()

    levenshtein = levenstein_(segments_pred, segments_true)
    assert levenshtein == levenstein_(segments_true, segments_pred)

    levenshtein_normalized = levenshtein / max(len(segments_pred), len(segments_true))
    assert 0 <= levenshtein_normalized <= 1

    return 1 - levenshtein_normalized


def calc_classification_scores(score_fn, y_true, y_pred, nlabels, nan_strategy="A"):
    assert score_fn in [precision_score, recall_score, f1_score, jaccard_score]
    assert nan_strategy in ["A", "B", "C"]

    labels = list(range(nlabels))
    scores = score_fn(y_true, y_pred, labels=labels, average=None, zero_division=0)
    assert len(scores) == nlabels

    for label in range(nlabels):
        label_missing = np.sum(y_true == label) == 0
        label_not_predicted = np.sum(y_pred == label) == 0

        if nan_strategy == "A":
            if score_fn == precision_score and label_not_predicted:
                scores[label] = np.nan
            elif score_fn == recall_score and label_missing:
                scores[label] = np.nan
            elif (score_fn in [f1_score, jaccard_score]) and (label_missing and label_not_predicted):
                scores[label] = np.nan
        elif nan_strategy == "B":
            if label_missing:
                scores[label] = np.nan
            elif score_fn == precision_score and label_not_predicted:
                scores[label] = np.nan
        else:  # C
            if label_missing:
                if score_fn == recall_score:
                    scores[label] = np.nan
                else:
                    if label_not_predicted:
                        scores[label] = 1
            elif score_fn == precision_score and label_not_predicted:
                scores[label] = np.nan

    return scores


def calc_video_metrics(y_true, y_pred, nlabels):
    if y_true.shape[0] != y_pred.shape[0]:
        raise RuntimeError(f"Incompatible shapes: y_true {y_true.shape}, y_pred {y_pred.shape}")

    metrics = {
        "accuracy": None,
        "balanced_accuracy": None,
        "confusion_matrix": None,
        "precision": {},
        "recall": {},
        "f1": {},
        "jaccard": {},
        "macro_precision": {},
        "macro_recall": {},
        "macro_f1": {},
        "macro_jaccard": {},
        "inflated_macro_f1": {},
        "edit_score": None,
    }

    labels = list(range(nlabels))

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred, adjusted=False)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels, normalize=None)

    for nan_strategy in ["A", "B"]:
        precision = calc_classification_scores(precision_score, y_true, y_pred, nlabels, nan_strategy=nan_strategy)
        macro_precision = np.nanmean(precision)

        recall = calc_classification_scores(recall_score, y_true, y_pred, nlabels, nan_strategy=nan_strategy)
        macro_recall = np.nanmean(recall)

        f1 = calc_classification_scores(f1_score, y_true, y_pred, nlabels, nan_strategy=nan_strategy)
        jaccard = calc_classification_scores(jaccard_score, y_true, y_pred, nlabels, nan_strategy=nan_strategy)

        metrics["precision"][nan_strategy] = precision
        metrics["macro_precision"][nan_strategy] = macro_precision
        metrics["recall"][nan_strategy] = recall
        metrics["macro_recall"][nan_strategy] = macro_recall
        metrics["f1"][nan_strategy] = f1
        metrics["macro_f1"][nan_strategy] = np.nanmean(f1)
        metrics["jaccard"][nan_strategy] = jaccard
        metrics["macro_jaccard"][nan_strategy] = np.nanmean(jaccard)

        if (macro_precision + macro_recall) == 0:
            metrics["inflated_macro_f1"][nan_strategy] = 0
        else:
            metrics["inflated_macro_f1"][nan_strategy] = (
                2 * macro_precision * macro_recall
            ) / (macro_precision + macro_recall)

    metrics["edit_score"] = calc_edit_score(y_true, y_pred)
    for overlap in [10, 25, 50, 75, 90]:
        metrics[f"overlap_f1_{overlap}"] = calc_overlap_f1(y_true, y_pred, nlabels, overlap=overlap / 100)

    return metrics


def calc_conf_mat_metrics(confusion_matrix_, nlabels):
    if confusion_matrix_.dtype != float:
        confusion_matrix_ = confusion_matrix_.astype(float)

    precision = np.zeros(nlabels, dtype=float)
    recall = np.zeros(nlabels, dtype=float)
    jaccard = np.zeros(nlabels, dtype=float)
    f1 = np.zeros(nlabels, dtype=float)

    for p in range(nlabels):
        if np.sum(confusion_matrix_[:, p]) == 0:
            precision[p] = np.nan
        else:
            precision[p] = confusion_matrix_[p, p] / np.sum(confusion_matrix_[:, p])

        if np.sum(confusion_matrix_[p, :]) == 0:
            recall[p] = np.nan
        else:
            recall[p] = confusion_matrix_[p, p] / np.sum(confusion_matrix_[p, :])

        if (np.sum(confusion_matrix_[p, :]) + np.sum(confusion_matrix_[:, p])) == 0:
            f1[p] = np.nan
        else:
            f1[p] = (2 * confusion_matrix_[p, p]) / (
                np.sum(confusion_matrix_[p, :]) + np.sum(confusion_matrix_[:, p])
            )

            if not (np.isnan(precision[p]) or np.isnan(recall[p]) or (precision[p] + recall[p] == 0)):
                f1_ = (2 * precision[p] * recall[p]) / (precision[p] + recall[p])
                assert abs(f1[p] - f1_) < eps

        denom = np.sum(confusion_matrix_[p, :]) + np.sum(confusion_matrix_[:, p]) - confusion_matrix_[p, p]
        if denom == 0:
            jaccard[p] = np.nan
        else:
            jaccard[p] = confusion_matrix_[p, p] / denom

    framewise_accuracy = np.trace(confusion_matrix_) / np.sum(confusion_matrix_)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "accuracy": framewise_accuracy,
    }


# -----------------------------
# Report summarization
# -----------------------------
def summarize_videowise_metric(experiment_results, test_ids, nruns, metric, nan_strategy=None):
    if nan_strategy is None:
        all_results = np.asarray(
            [[experiment_results[run][video_id][metric] for video_id in test_ids] for run in range(nruns)]
        )
    else:
        all_results = np.asarray(
            [[experiment_results[run][video_id][metric][nan_strategy] for video_id in test_ids] for run in range(nruns)]
        )

    return {
        "mean": np.mean(all_results).item(),
        "std_V": np.std(np.mean(all_results, axis=0), ddof=1).item(),
        "std_R": np.std(np.mean(all_results, axis=1), ddof=1).item() if nruns > 1 else np.nan,
    }


def summarize_phasewise_videowise_metric(experiment_results, test_ids, nruns, metric, nlabels, nan_strategy=None):
    result = {}

    if nan_strategy is None:
        all_results = np.asarray(
            [[experiment_results[run][video_id][metric] for video_id in test_ids] for run in range(nruns)]
        )
    else:
        all_results = np.asarray(
            [[experiment_results[run][video_id][metric][nan_strategy] for video_id in test_ids] for run in range(nruns)]
        )

    result["mean"] = np.nanmean(all_results).item()
    result["std_V"] = np.std(np.nanmean(all_results, axis=(0, 2)), ddof=1).item()
    result["std_P"] = np.std(np.nanmean(all_results, axis=(0, 1)), ddof=1).item()
    result["std_R"] = np.std(np.nanmean(all_results, axis=(1, 2)), ddof=1).item()

    for phase in range(nlabels):
        phase_results = all_results[:, :, phase]

        mean_per_video = np.nanmean(phase_results, axis=0)
        n_valid = np.sum(~np.isnan(mean_per_video))
        if n_valid == 0:
            _std_V_ = np.nan
        elif n_valid == 1:
            _std_V_ = 0
        else:
            _std_V_ = np.nanstd(mean_per_video, ddof=1).item()

        result[phase] = {
            "mean": np.nanmean(phase_results).item(),
            "std_V": _std_V_,
            "std_R": np.nanstd(np.nanmean(phase_results, axis=1), ddof=1).item() if nruns > 1 else np.nan,
        }

    std_P_ = np.std([result[phase]["mean"] for phase in range(nlabels)], ddof=1).item()
    if abs(result["std_P"] - std_P_) >= eps:
        warnings.warn(
            f"{metric}[{nan_strategy}]: Unexpectedly high deviation ({eps}) between "
            f"std_P ({result['std_P']}) and std of phase-wise means ({std_P_})."
        )

    result["mean_P"] = np.mean([result[phase]["mean"] for phase in range(nlabels)]).item()
    return result


def calculate_f1_upper_bound(experiment_results, test_ids, nruns, nlabels, nan_strategy):
    result = {}

    precision_scores = np.asarray(
        [[experiment_results[run][video_id]["precision"][nan_strategy] for video_id in test_ids] for run in range(nruns)]
    )
    recall_scores = np.asarray(
        [[experiment_results[run][video_id]["recall"][nan_strategy] for video_id in test_ids] for run in range(nruns)]
    )

    mean_precision_per_run = np.nanmean(precision_scores, axis=(1, 2))
    mean_recall_per_run = np.nanmean(recall_scores, axis=(1, 2))

    f1_upper_bound_per_run = [
        (2 * mean_precision_per_run[i] * mean_recall_per_run[i]) / (mean_precision_per_run[i] + mean_recall_per_run[i])
        for i in range(nruns)
    ]

    result["mean"] = np.mean(f1_upper_bound_per_run).item()
    result["std_R"] = np.std(f1_upper_bound_per_run, ddof=1).item() if nruns > 1 else np.nan

    if nruns > 1:
        mean_precision = np.nanmean(precision_scores)
        mean_recall = np.nanmean(recall_scores)
        result["overall"] = ((2 * mean_precision * mean_recall) / (mean_precision + mean_recall)).item()

    return result


def summarize_framewise_metric(experiment_results, nruns, metric, nlabels):
    result = {}

    if metric != "accuracy":
        all_results = np.asarray([experiment_results[run][f"framewise_{metric}"] for run in range(nruns)])

        if metric == "precision":
            result["mean"] = np.nanmean(all_results).item()
            result["std_P"] = np.std(np.nanmean(all_results, axis=0), ddof=1).item()
            result["std_R"] = np.std(np.nanmean(all_results, axis=1), ddof=1).item() if nruns > 1 else np.nan

            for phase in range(nlabels):
                _results = all_results[:, phase]
                result[phase] = {
                    "mean": np.nanmean(_results).item(),
                    "std_R": np.nanstd(_results, ddof=1).item() if nruns > 1 else np.nan,
                }
        else:
            result["mean"] = np.mean(all_results).item()
            result["std_P"] = np.std(np.mean(all_results, axis=0), ddof=1).item()
            result["std_R"] = np.std(np.mean(all_results, axis=1), ddof=1).item() if nruns > 1 else np.nan

            for phase in range(nlabels):
                _results = all_results[:, phase]
                result[phase] = {
                    "mean": np.mean(_results).item(),
                    "std_R": np.std(_results, ddof=1).item() if nruns > 1 else np.nan,
                }
    else:
        all_results = np.asarray([experiment_results[run]["framewise_accuracy"] for run in range(nruns)])
        result["mean"] = np.mean(all_results).item()
        result["std_R"] = np.std(all_results, ddof=1).item() if nruns > 1 else np.nan

    return result


# -----------------------------
# High-level evaluation
# -----------------------------
def get_results_from_txt_dir(predictions_dir, nlabels):
    videos = load_all_video_predictions(predictions_dir)
    test_ids = sorted(videos.keys())

    run_results = {}
    framewise_confusion_matrix = np.zeros([nlabels, nlabels], dtype=np.int64)

    for video_id in test_ids:
        y_true = videos[video_id]["y_true"]
        y_pred = videos[video_id]["y_pred"]

        if y_pred.shape[0] == (y_true.shape[0] - 1):
            expected_len = y_true.shape[0]
            y_true = y_true[:-1]
            warnings.warn(
                f"Predictions for video {video_id} have length {y_pred.shape[0]}, "
                f"but expected length {expected_len}."
            )

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Cannot compare predictions for video {video_id} with length {y_pred.shape[0]} "
                f"to ground truth with length {y_true.shape[0]}."
            )

        run_results[video_id] = calc_video_metrics(y_true, y_pred, nlabels)
        framewise_confusion_matrix += run_results[video_id]["confusion_matrix"]

    framewise_metrics = calc_conf_mat_metrics(framewise_confusion_matrix, nlabels)
    for metric in framewise_metrics:
        run_results[f"framewise_{metric}"] = framewise_metrics[metric]

    run_results["run"] = (0, "by_video")
    return [run_results], test_ids


def infer_nlabels(videos):
    max_label = -1
    for _, vp in videos.items():
        max_label = max(max_label, int(np.max(vp["y_true"])), int(np.max(vp["y_pred"])))
    return max_label + 1


def report_results(
    predictions_dir,
    out_dir,
    nlabels=None,
    force_overwrite=False,
    plot_predictions=False,
    plots_dir=None,
):
    predictions_dir = os.fspath(predictions_dir)
    out_dir = os.fspath(out_dir)
    plots_dir = None if plots_dir is None else os.fspath(plots_dir)

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "eval.yaml")

    if os.path.exists(out_file) and not force_overwrite:
        print(f"Found evaluation report at '{out_file}'.")
        return yaml.safe_load(open(out_file))

    videos = load_all_video_predictions(predictions_dir)
    if nlabels is None:
        nlabels = infer_nlabels(videos)

    experiment_results, test_ids = get_results_from_txt_dir(predictions_dir, nlabels)
    nruns = len(experiment_results)

    report = {
        "predictions_dir": predictions_dir,
        "run_dirs": ["by_video"],
        "fps": 1,
        "nlabels": int(nlabels),
        "video_ids": test_ids,
    }

    for metric in ["accuracy", "balanced_accuracy", "edit_score"] + [f"overlap_f1_{o}" for o in [10, 25, 50, 75, 90]]:
        report[metric] = summarize_videowise_metric(experiment_results, test_ids, nruns, metric)

    for metric in ["macro_precision", "macro_recall", "macro_f1", "macro_jaccard", "inflated_macro_f1"]:
        report[metric] = {}
        for nan_strategy in ["A", "B"]:
            report[metric][nan_strategy] = summarize_videowise_metric(
                experiment_results, test_ids, nruns, metric, nan_strategy
            )

    for metric in ["precision", "recall", "f1", "jaccard"]:
        report[metric] = {}
        for nan_strategy in ["A", "B"]:
            report[metric][nan_strategy] = summarize_phasewise_videowise_metric(
                experiment_results, test_ids, nruns, metric, nlabels, nan_strategy
            )

    report["inflated_macro_f1_upper_bound"] = {}
    for nan_strategy in ["A", "B"]:
        report["inflated_macro_f1_upper_bound"][nan_strategy] = calculate_f1_upper_bound(
            experiment_results, test_ids, nruns, nlabels, nan_strategy
        )

    for metric in ["accuracy", "precision", "recall", "f1", "jaccard"]:
        report[f"framewise_{metric}"] = summarize_framewise_metric(experiment_results, nruns, metric, nlabels)

    if plot_predictions:
        if plots_dir is None:
            plots_dir = os.path.join(out_dir, "plots")
        saved_pngs = plot_predictions_from_txt_dir(
            predictions_dir=predictions_dir,
            plots_dir=plots_dir,
            nlabels=nlabels,
        )
        report["plots_dir"] = str(plots_dir)
        report["num_plots"] = len(saved_pngs)

    yaml.safe_dump(report, stream=open(out_file, "w"), default_flow_style=False)
    print(f"Stored evaluation report at '{out_file}'.")
    if plot_predictions:
        print(f"Stored {report['num_plots']} prediction plot(s) at '{report['plots_dir']}'.")

    # Save detailed per-video results to Excel
    export_to_excel(experiment_results[0], test_ids, nlabels, out_dir)

    return report


def export_to_excel(video_results, video_ids, nlabels, out_dir):
    """Save all metrics per video to a multi-sheet Excel file."""
    excel_path = os.path.join(out_dir, "eval_results.xlsx")
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")

    # 1. Simple Metrics (Accuracy, Balanced Accuracy, Edit Score, Overlaps)
    simple_metrics = ["accuracy", "balanced_accuracy", "edit_score"] + [
        f"overlap_f1_{o}" for o in [10, 25, 50, 75, 90]
    ]
    data_simple = []
    for vid in video_ids:
        row = {"video_id": vid}
        for m in simple_metrics:
            row[m] = video_results[vid][m]
        data_simple.append(row)
    pd.DataFrame(data_simple).to_excel(writer, sheet_name="General Metrics", index=False)

    # 2. Per-class metrics (F1, Precision, Recall, Jaccard)
    # We use Nan Strategy 'A' as the primary one for individual video details
    class_metrics = ["f1", "precision", "recall", "jaccard"]
    for m in class_metrics:
        data_class = []
        for vid in video_ids:
            row = {"video_id": vid}
            scores = video_results[vid][m]["A"]  # Using strategy A
            for i, score in enumerate(scores):
                row[f"Class_{i}"] = score
            # Add Macro mean for this video
            row["Macro_Mean"] = np.nanmean(scores)
            data_class.append(row)
        pd.DataFrame(data_class).to_excel(writer, sheet_name=m.capitalize(), index=False)

    writer.close()
    print(f"Detailed per-video results saved to '{excel_path}'.")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Evaluate paired GT/pred txt files and write eval.yaml summary."
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="/projects/prjs1363/MTL/eomt_ramie_mtl/inference_outputs/classification/by_video",
        help="Directory containing per-video txt files with columns: frame, gt, pred, conf",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where eval.yaml will be written.",
    )
    parser.add_argument(
        "--nlabels",
        type=int,
        default=None,
        help="Total number of phase labels. If omitted, inferred from max label in GT/pred.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing eval.yaml in output-dir.",
    )
    parser.add_argument(
        "--plot-predictions",
        action="store_true",
        help="If set, generate one PNG plot per prediction txt file.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory to write prediction plots. Defaults to <output-dir>/plots.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    report_results(
        predictions_dir=args.predictions_dir,
        out_dir=args.output_dir,
        nlabels=args.nlabels,
        force_overwrite=args.force_overwrite,
        plot_predictions=args.plot_predictions,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()
