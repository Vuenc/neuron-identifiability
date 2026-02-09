import train
import argparse
import hydra
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate.py")
    parser.add_argument("run_directory")
    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path=args.run_directory):
        cfg = hydra.compose(config_name="config")

    # # Attach new wandb run to old run's group
    # if cfg.logging.get('use_wandb', False):
    #     import wandb
    #     previous_run_id = args.run_directory[args.run_directory.rfind("_")+1:]
    #     api = wandb.Api()
    #     previous_run = api.run(f"asymmetric-networks/asymmetric-networks/{previous_run_id}")
    #     cfg.logging.group = previous_run.group
    #     print(previous_run.group)

    wandb = train.setup_wandb(cfg)

    mask_seed = cfg.mask_seed
    if mask_seed is None:
        mask_seed = -1
        print("Warning: using new mask seed. This shouldn't matter for evaluation, but I'm not sure.")

    train.evaluate_all(cfg, Path(args.run_directory), mask_seed=mask_seed)