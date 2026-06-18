import argparse
import os
import time
import random
import numpy as np
import torch
import util


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        # TODO: set default values
        parser.add_argument('--dataroot', default=None,
                            help='Training data root directory')
        # TODO: set default values
        parser.add_argument('--test_dataroot', default=None,
                            help='Test data root directory, defaults to dataroot')
        parser.add_argument('--sample_list', default=None, type=str,
                            help='Optional path to a sample list file for deterministic ordering')
        parser.add_argument('--classes', default='car,cat,chair,horse,sdv1.4')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--loadSize', type=int, default=256)
        parser.add_argument('--cropSize', type=int, default=224)
        parser.add_argument('--num_threads', default=4, type=int)
        
        parser.add_argument('--name', type=str, default='experiment')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        parser.add_argument('--gpu_id', type=int, default=1, help='GPU id to use (single GPU mode)')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        
        # Multi-GPU
        parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
        parser.add_argument('--world_size', type=int, default=1, help='Total number of GPUs')
        parser.add_argument('--rank', type=int, default=0, help='Current process rank')
        parser.add_argument('--local_rank', type=int, default=-1, help='Local GPU rank (set by torch.distributed.launch)')
        parser.add_argument('--dist_url', type=str, default='env://', help='Distributed init URL')
        parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
        
        parser.add_argument('--delr_freq', type=int, default=20)
        
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--no_resize', action='store_true')
        
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return opt

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        if opt.name == 'experiment':
            opt.name = opt.name + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

        self.set_seed(opt.seed)

        if hasattr(opt, 'local_rank') and opt.local_rank >= 0:
            opt.distributed = True
            opt.rank = opt.local_rank
            opt.gpu_id = opt.local_rank
            torch.cuda.set_device(opt.local_rank)
        elif opt.distributed:
            if opt.local_rank >= 0:
                opt.gpu_id = opt.local_rank
                torch.cuda.set_device(opt.local_rank)
        else:
            if opt.gpu_id >= 0 and torch.cuda.is_available():
                torch.cuda.set_device(opt.gpu_id)

        opt.classes = opt.classes.split(',')

        if opt.test_dataroot is None:
            opt.test_dataroot = opt.dataroot

        self.opt = opt
        return self.opt
    
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
