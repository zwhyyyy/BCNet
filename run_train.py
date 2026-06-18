"""
Single-experiment training launcher with full reproducibility.

All parameters are read from options/ (TrainOptions + BaseOptions).

Usage:
    # Train with 3 GPUs, all defaults from options/:
    python run_train.py --nproc_per_node 3

    # Override any training param on the command line:
    python run_train.py --nproc_per_node 3 --ase_threshold 0.5

    # Dry run (print command without executing):
    python run_train.py --nproc_per_node 3 --dry_run
"""

import os
import sys
import subprocess
import time
import shutil

from options.train_options import TrainOptions


def main():
    opt = TrainOptions().parse()

    cmd, exp_dir = _build_train_cmd(opt)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Single-Experiment Training  (all params from options/)")
    print(f"{sep}")
    print(f"  experiment name  : {opt.name}")
    print(f"  nproc_per_node   : {opt.nproc_per_node}")
    print(f"  dataroot         : {opt.dataroot}")
    print(f"  checkpoints_dir  : {opt.checkpoints_dir}")
    print(f"  output dir       : {exp_dir}")
    print(f"  command          : {' '.join(cmd)}")
    print(f"{sep}\n")

    if opt.dry_run:
        print("[DRY RUN -- not executed]")
        return

    t0 = time.time()
    rc, err = _run_training(cmd, exp_dir)
    duration = time.time() - t0

    print(f"\n{sep}")
    if rc == 0:
        print(f"  SUCCESS  ({duration:.0f}s)")
        print(f"  checkpoint dir: {exp_dir}")
    else:
        print(f"  FAILED  (exit code {rc})")
        print(f"  see: {os.path.join(exp_dir, 'log.txt')}")
        if err:
            print(f"  error: {err}")
    print(f"{sep}\n")


# ============================================================================
#  Build command
# ============================================================================

def _build_train_cmd(opt):
    exp_dir = os.path.join(opt.checkpoints_dir, opt.name)

    torchrun_path = shutil.which("torchrun")
    if torchrun_path is None:
        raise RuntimeError(
            "torchrun not found in PATH. "
            "Please ensure PyTorch is installed and torchrun is accessible."
        )

    cmd = [
        torchrun_path,
        f"--nproc_per_node={opt.nproc_per_node}",
        "train.py",
        "--dataroot",        opt.dataroot,
        "--checkpoints_dir", opt.checkpoints_dir,
        "--name",            opt.name,
        "--batch_size",      str(opt.batch_size),
        "--loadSize",        str(opt.loadSize),
        "--cropSize",        str(opt.cropSize),
        "--num_threads",     str(opt.num_threads),
        "--niter",           str(opt.niter),
        "--lr",              str(opt.lr),
        "--weight_decay",    str(opt.weight_decay),
        "--save_epoch_freq", str(opt.save_epoch_freq),
        "--log_freq",        str(opt.log_freq),
        "--lora_r",          str(opt.lora_r),
        "--lora_alpha",      str(opt.lora_alpha),
        "--lora_dropout",    str(opt.lora_dropout),
        "--seed",            str(opt.seed),
        "--ase_threshold",   str(opt.ase_threshold),
        "--npe_epsilon",     str(opt.npe_epsilon),
        "--ase_loss_weight", str(opt.ase_loss_weight),
        "--npe_loss_weight", str(opt.npe_loss_weight),
        "--clean_loss_weight", str(opt.clean_loss_weight),
    ]

    return cmd, exp_dir


# ============================================================================
#  Subprocess execution
# ============================================================================

def _clean_env():
    """Strip torchrun/distributed env vars so the subprocess starts fresh."""
    env = os.environ.copy()
    for key in [
        "RANK", "WORLD_SIZE", "LOCAL_RANK",
        "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
        "GROUP_RANK", "ROLE_RANK", "ROLE_NAME",
        "MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE",
    ]:
        env.pop(key, None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def _run_training(cmd, exp_dir):
    """Run training subprocess; stream output to console and exp_dir/log.txt."""
    env = _clean_env()
    log_file = os.path.join(exp_dir, "log.txt")
    os.makedirs(exp_dir, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait()

    rc = proc.returncode
    err = None
    if rc != 0:
        try:
            with open(log_file, encoding="utf-8") as lf:
                lines = lf.readlines()
            err = "".join(lines[-5:]).strip()
        except Exception:
            err = f"exit code {rc}"

    return rc, err


if __name__ == "__main__":
    main()
