'''
This script is for plotting the per-phase results for multiple models in radar plot.
Experiment_dir:
/projects/prjs1363/SurgPhaseBench/outputs/

'''

Experiments = {
    "SV-RCNet": "/projects/prjs1363/SurgPhaseBench/outputs/20260413_End2End_ResNet50LSTM_split1_ramie/test_results",
    "TMRNet": "/projects/prjs1363/SurgPhaseBench/outputs/20260414_tmrnet_resnet50_ramie_split1/test_results",
    "TeCNO": "/projects/prjs1363/SurgPhaseBench/outputs/20260412_Stage3_MSTCN_resnet50_RAMIE_split1/test_results",
    "Trans-SVNet": "/projects/prjs1363/SurgPhaseBench/outputs/20260414_temporal_trans_svnet_resnet50_RAMIE_split1/test_results",
    "Causal-Transformer": "/projects/prjs1363/SurgPhaseBench/outputs/20260412_Stage3_ASFormer_Causal_ResNet50_RAMIE_split1/test_results",
    "DINO+TeCNO": "/projects/prjs1363/SurgPhaseBench/outputs/20260412_Stage3_MSTCN_DINOv2_RAMIE_split1/test_results",
    "DINO+Causal-Transformer": "/projects/prjs1363/SurgPhaseBench/outputs/20260412_Stage3_ASFormer_Causal_DINOv2_RAMIE_split1/test_results",
    }

import argparse
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle

def read_eval_yaml(experiment_path):
    yaml_path = os.path.join(experiment_path, 'eval.yaml')
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def create_radar_plot(data, metric, experiments_dict, output_path, calc_method=None):
    

    custom_colors = [
    "#FFE119",  # 0 Vivid Yellow
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

    "#000000",  # 13 Black
    ]
    
    # phase is from 0 to 12
    phases = list(range(13))
    phase_names = list(range(1,14))
    
    num_vars = len(phases)
    print(f"Number of phases: {num_vars}")
    
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for legend_name, path in experiments_dict.items():
        # check if 'A' or 'mean' is the key
        if calc_method and calc_method in data[legend_name][metric]:
             values = [data[legend_name][metric][calc_method][phase]['mean'] for phase in phases]
        elif 'A' in data[legend_name][metric]:
            calculation_method = 'A'
            print(f"Using calculation method 'A' for {legend_name}")
            values = [data[legend_name][metric][calculation_method][phase]['mean'] for phase in phases]
        elif 'B' in data[legend_name][metric]:
             calculation_method = 'B'
             values = [data[legend_name][metric][calculation_method][phase]['mean'] for phase in phases]
        else:
            values = [data[legend_name][metric][phase]['mean'] for phase in phases]
        
        print(f"Values for {legend_name}: {values}")
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=legend_name)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(phase_names)
    
    # Add a color for the phase_names
    color_palette = sns.color_palette(custom_colors[:len(phase_names)])
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color('black')
        label.set_fontsize(30)
        angle = angles[i]
        bbox_props = dict(boxstyle="round,pad=0.2", facecolor=color_palette[i], edgecolor="none")
        label.set_bbox(bbox_props)
    
    ax.tick_params(axis='x', labelsize=30)
    ax.set_ylim(0, 1)  
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis='y', labelsize=25)

    ax.legend(loc='upper right', bbox_to_anchor=(1.65, 1.1), fontsize=20)
    
    plt.title(f"F1 score across phases", fontsize=30, loc='center', pad=20)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_charts(experiments_dict, metrics, output_dir, filetype, calc_method=None):
    data = {}
    for legend_name, path in experiments_dict.items():
        print(f"Reading data from {path}")
        data[legend_name] = read_eval_yaml(path)

    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        output_path = os.path.join(output_dir, f"{metric}_radar.{filetype}")
        create_radar_plot(data, metric, experiments_dict, output_path, calc_method)


if __name__ == "__main__":
    
    metrics_to_plot = ["f1"] # Default metric
    output_directory = "radar_plots"
    file_extension = "png"
    calculation_method = "A" # Can be customized here

    plot_radar_charts(Experiments, metrics_to_plot, output_directory, file_extension, calculation_method)


