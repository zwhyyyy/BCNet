import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from datetime import timedelta

from options.train_options import TrainOptions
from data.datasets import dataset_folder
from networks.trainer import Trainer
import util


class Logger:
    """Output to both console and file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def save_options(opt, save_path):
    """Save options to a txt file."""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('----------------- Options ---------------\n')
        for k, v in sorted(vars(opt).items()):
            f.write(f'{k}: {v}\n')
        f.write('----------------- End -------------------\n')


def init_distributed(opt):
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ['RANK'])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.local_rank = int(os.environ.get('LOCAL_RANK', opt.rank))
    elif opt.local_rank >= 0:
        opt.rank = opt.local_rank
        opt.world_size = torch.cuda.device_count()
    else:
        opt.distributed = False
        return False
    
    opt.distributed = opt.world_size > 1
    
    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        
        timeout = timedelta(minutes=30)
        
        dist.init_process_group(
            backend=opt.dist_backend,
            init_method=opt.dist_url,
            world_size=opt.world_size,
            rank=opt.rank,
            timeout=timeout
        )
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        if opt.rank == 0:
            print(f"Distributed training initialized: rank={opt.rank}, world_size={opt.world_size}")
            print(f"NCCL timeout set to {timeout}")
        return True
    return False


def train():
    opt = TrainOptions().parse()
    
    is_distributed = init_distributed(opt)
    
    if not is_distributed or opt.rank == 0:
        save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(save_dir)
        save_options(opt, os.path.join(save_dir, 'params.txt'))
        sys.stdout = Logger(os.path.join(save_dir, 'log.txt'))
    else:
        save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(save_dir)
    
    train_path = os.path.join(opt.dataroot, opt.train_split)
    if not os.path.exists(train_path):
        raise ValueError(f"No training data found in {train_path}")
    
    train_dataset = dataset_folder(opt, train_path)
    
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank,
            shuffle=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=opt.batch_size, 
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=opt.num_threads, 
        pin_memory=True, 
        drop_last=True
    )
    
    if opt.rank == 0 or not is_distributed:
        print(f"Training samples: {len(train_dataset)}")
        world_size = getattr(opt, 'world_size', 1)
        print(f"Batch size per GPU: {opt.batch_size}, Total batch size: {opt.batch_size * world_size}")
    
    trainer = Trainer(opt)
    
    for epoch in range(opt.epoch_count, opt.niter + 1):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        trainer.train_epoch(train_loader, epoch)
        
        if is_distributed:
            dist.barrier()
        
        if opt.rank == 0:
            print(f'\n--- Epoch {epoch} Evaluation ---')
        
        results = trainer.evaluate()
        
        if opt.rank == 0 or not is_distributed:
            avg_acc = np.mean([r['acc'] for r in results.values()]) if results else 0
            trainer.update_best(avg_acc, epoch)
            trainer.save_model(epoch)
        
        if is_distributed:
            dist.barrier()
        
        trainer.step_scheduler()
    
    if opt.rank == 0 or not is_distributed:
        print(f'Training finished. Best avg acc: {trainer.best_avg_acc:.2f}%')
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    train()
