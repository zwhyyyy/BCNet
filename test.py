import os
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, average_precision_score

from options.test_options import TestOptions
from data.datasets import dataset_folder
from models.dino_model import create_model


# AIGI-Bench dataset classes
VALS = ['BLIP', 'BlendFace', 'CommunityAI', 'DALLE-3', 'E4S', 'FLUX1-dev', 
        'FaceSwap', 'GLIDE', 'Imagen3', 'InSwap', 'InstantID', 'Midjourney', 
        'PhotoMaker', 'ProGAN', 'R3GAN', 'SD3', 'SDXL', 'SimSwap', 'SocialRF', 
        'StyleGAN-XL', 'StyleGAN3', 'StyleSwim', 'WFIR', 'Infinite_ID', 'IP_Adapter']


def load_model(opt, device):
    print("Creating model...")
    model = create_model(opt)
    
    print(f"Loading checkpoint from: {opt.model_path}")
    if os.path.exists(opt.model_path):
        checkpoint = torch.load(opt.model_path, map_location='cpu', weights_only=False)

        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'lora_state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['lora_state_dict'], strict=False)
            print(f"✓ Loaded LoRA weights from {opt.model_path}")
            if missing_keys:
                print(f"  Missing keys (expected, backbone frozen): {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys}")
        elif 'adapter_state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['adapter_state_dict'], strict=False)
            print(f"✓ Loaded Adapter weights from {opt.model_path}")
        elif 'model_state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✓ Loaded full model from {opt.model_path}")
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            print(f"✓ Loaded weights from {opt.model_path}")

        if 'epoch' in checkpoint:
            print(f"  Model trained for {checkpoint['epoch']} epochs")
        if 'best_acc' in checkpoint:
            print(f"  Best accuracy: {checkpoint['best_acc']:.2f}%")
    else:
        raise FileNotFoundError(f"Model file not found: {opt.model_path}")
    
    model = model.to(device)
    model.eval()
    print("Model loaded and set to eval mode")
    return model


def evaluate(model, opt, device, rank=0, world_size=1, is_distributed=False):
    results = {}

    if is_distributed:
        val_names_per_gpu = [VALS[i] for i in range(rank, len(VALS), world_size)]
        if rank == 0:
            print(f"Distributed evaluation: GPU {rank} will evaluate {len(val_names_per_gpu)} datasets: {val_names_per_gpu}")
    else:
        val_names_per_gpu = VALS
    
    for val_name in val_names_per_gpu:
        test_path = os.path.join(opt.test_dataroot, val_name)
        if not os.path.exists(test_path):
            if rank == 0 or not is_distributed:
                print(f'Warning: Test path not found: {test_path}')
            continue
        
        test_dataset = dataset_folder(opt, test_path)

        if is_distributed:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            test_sampler = None
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=opt.batch_size, 
            shuffle=False,
            sampler=test_sampler,
            num_workers=opt.num_threads, 
            pin_memory=True
        )
        
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = (all_probs > 0.5).astype(int)
        
        acc = accuracy_score(all_labels, all_preds) * 100
        if len(np.unique(all_labels)) > 1:
            ap = average_precision_score(all_labels, all_probs) * 100
        else:
            ap = acc
        
        results[val_name] = {'acc': acc, 'ap': ap}
        print(f'[GPU {rank}] {val_name}: Acc={acc:.2f}%, AP={ap:.2f}%')

    if is_distributed:
        all_results_list = [None] * world_size
        dist.all_gather_object(all_results_list, results)
        
        merged_results = {}
        for r in all_results_list:
            merged_results.update(r)
        
        return merged_results
    
    return results


def init_distributed_test(opt):
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
        dist.init_process_group(
            backend=opt.dist_backend,
            init_method=opt.dist_url,
            world_size=opt.world_size,
            rank=opt.rank
        )
        if opt.rank == 0:
            print(f"Distributed testing initialized: rank={opt.rank}, world_size={opt.world_size}")
        return True
    return False


def test():
    opt = TestOptions().parse()

    if not os.path.exists(opt.model_path):
        raise FileNotFoundError(f"Model file not found: {opt.model_path}")
    
    print(f"Loading model from: {opt.model_path}")

    is_distributed = init_distributed_test(opt)

    if is_distributed:
        device = torch.device(f'cuda:{opt.local_rank}')
        rank = opt.rank
        world_size = opt.world_size
    else:
        device = torch.device(f'cuda:{opt.gpu_id}' if opt.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
        rank = 0
        world_size = 1
    
    print(f"Using device: {device}")

    model = load_model(opt, device)
    

    if rank == 0 or not is_distributed:
        print('\n--- Evaluation ---')
    
    results = evaluate(model, opt, device, rank=rank, world_size=world_size, is_distributed=is_distributed)

    if rank == 0 or not is_distributed:
        avg_acc = np.mean([r['acc'] for r in results.values()])
        avg_ap = np.mean([r['ap'] for r in results.values()])
        print(f'\nAverage: Acc={avg_acc:.2f}%, AP={avg_ap:.2f}%')

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    test()
