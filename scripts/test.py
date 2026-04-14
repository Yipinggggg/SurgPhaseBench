import sys
from pathlib import Path
import yaml
import pytorch_lightning as pl
import argparse

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.tasks.end_to_end_module import EndToEndModule
from src.tasks.tmrnet_module import TMRNetModule
from src.data.datamodules.end_to_end_sequence_datamodule import EndToEndSequenceDataModule
from src.data.datamodules.tmrnet_datamodule import TMRNetDataModule

def main():
    parser = argparse.ArgumentParser(description="Re-run test on existing checkpoint")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    mode = cfg.get("mode", "end_to_end")
    
    # # seq_stride can be overridden at test time to ensure consistent evaluation across models
    # if cfg.get("data", {}).get("seq_stride", 1) != 1:
    #     print(f"Overriding seq_stride to 1 for testing (was {cfg['data']['seq_stride']})")
    #     cfg["data"]["seq_stride"] = 1

    if "tmrnet" in mode.lower():
        datamodule = TMRNetDataModule(cfg)
        module = TMRNetModule.load_from_checkpoint(args.checkpoint, cfg=cfg)
    else:
        datamodule = EndToEndSequenceDataModule(cfg)
        module = EndToEndModule.load_from_checkpoint(args.checkpoint, cfg=cfg)
    
    trainer = pl.Trainer(default_root_dir=str(Path(args.checkpoint).parent.parent))
    trainer.test(module, datamodule=datamodule)

if __name__ == "__main__":
    main()