from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        # Training
        parser.add_argument('--train_split', type=str, default='train')
        parser.add_argument('--val_split', type=str, default='val')
        parser.add_argument('--epoch_count', type=int, default=1)
        parser.add_argument('--niter', type=int, default=1)
        parser.add_argument('--lr', type=float, default=0.00005)
        parser.add_argument('--weight_decay', type=float, default=0.001)
        parser.add_argument('--save_epoch_freq', type=int, default=1)
        parser.add_argument('--log_freq', type=int, default=100)
        
        # LoRA
        parser.add_argument('--lora_layers', type=str, default='all',
                            help='LoRA injection layers: all, last4, last8, last12, or comma-separated layer numbers')
        parser.add_argument('--lora_r', type=int, default=16,
                            help='LoRA rank, higher means more params but stronger fitting')
        parser.add_argument('--lora_alpha', type=int, default=32,
                            help='LoRA scaling factor')
        parser.add_argument('--lora_dropout', type=float, default=0.3,
                            help='LoRA dropout')
        
        # Adversarial erasure
        parser.add_argument('--use_adversarial_ase', action='store_true', default=True,
                            help='Enable adversarial erasure training')
        parser.add_argument('--ase_threshold', type=float, default=0.75,
                            help='Attention erasure threshold, regions above this are ased')
        parser.add_argument('--ase_loss_weight', type=float, default=0.45,
                            help='Erasure loss weight')
        
        # npe
        parser.add_argument('--npe_epsilon', type=float, default=0.0005,
                            help='npe perturbation strength')
        parser.add_argument('--npe_loss_weight', type=float, default=0.45,
                            help='npe adversarial loss weight')
        parser.add_argument('--clean_loss_weight', type=float, default=0.1,
                            help='Clean loss weight')
        
        # Launcher
        parser.add_argument('--nproc_per_node', type=int, default=1,
                            help='GPUs per experiment (torchrun --nproc_per_node)')
        parser.add_argument('--dry_run', action='store_true',
                            help='Print command without executing')
        
        self.isTrain = True
        return parser
