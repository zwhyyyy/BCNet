from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        parser.add_argument('--model_path', default="./checkpoint.pth", type=str,
                            help='Path to the trained model checkpoint')
        parser.set_defaults(sample_list=None)
        
        # LoRA (must match training config)
        parser.add_argument('--lora_layers', type=str, default='all',
                            help='LoRA injection layers: all, last4, last8, last12, or comma-separated layer numbers')
        parser.add_argument('--lora_r', type=int, default=16,
                            help='LoRA rank')
        parser.add_argument('--lora_alpha', type=int, default=32,
                            help='LoRA scaling factor')
        parser.add_argument('--lora_dropout', type=float, default=0.3,
                            help='LoRA dropout')
        
        self.isTrain = False
        return parser
