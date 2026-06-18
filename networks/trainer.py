import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score

from data.datasets import dataset_folder
from models.dino_model import create_model, generate_erasure_mask
import util


VALS = ['BLIP', 'BlendFace', 'CommunityAI', 'DALLE-3', 'E4S', 'FLUX1-dev', 
        'FaceSwap', 'GLIDE', 'Imagen3', 'InSwap', 'InstantID', 'Midjourney', 
        'PhotoMaker', 'ProGAN', 'R3GAN', 'SD3', 'SDXL', 'SimSwap', 'SocialRF', 
        'StyleGAN-XL', 'StyleGAN3', 'StyleSwim', 'WFIR', 'Infinite_ID', 'IP_Adapter']


def npe_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -2.5, 2.5)
    return perturbed_image


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.is_distributed = getattr(opt, 'distributed', False)
        self.rank = getattr(opt, 'rank', 0)
        # Setup device
        if self.is_distributed:
            self.device = torch.device(f'cuda:{opt.local_rank}')
        else:
            self.device = torch.device(f'cuda:{opt.gpu_id}' if opt.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
        
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if self.rank == 0 or not self.is_distributed:
            util.mkdirs(self.save_dir)
        
        self.model = create_model(opt)

        if self.rank == 0 or not self.is_distributed:
            model_obj = self.model.module if hasattr(self.model, "module") else self.model
            backbone = getattr(model_obj, "backbone", None)
            if backbone is not None:
                sample_weight = None
                for name, param in backbone.named_parameters():
                    if 'blocks.0.norm1.weight' in name:
                        sample_weight = param
                        break
                
                if sample_weight is not None:
                    weight_norm = sample_weight.norm().item()
                    if weight_norm > 0.01:
                        print(f"✓ Backbone weights verified (sample norm: {weight_norm:.4f})")
                    else:
                        print(f"⚠ Warning: Backbone weights may not be loaded (sample norm: {weight_norm:.4f})")
        
        # Move model to device
        self.model.to(self.device)
        
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=True
            )
            if self.rank == 0:
                print(f"Model wrapped with DistributedDataParallel")
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.use_adversarial_ase = getattr(opt, 'use_adversarial_ase', True)
        self.ase_threshold = getattr(opt, 'ase_threshold', 0.75)
        self.npe_epsilon = getattr(opt, 'npe_epsilon', 0.0005)
        
        # Loss weights: ase 0.45, npe 0.45, clean 0.1
        self.ase_loss_weight = getattr(opt, 'ase_loss_weight', 0.45)
        self.npe_loss_weight = getattr(opt, 'npe_loss_weight', 0.45)
        self.clean_loss_weight = getattr(opt, 'clean_loss_weight', 0.1)
        
        if self.use_adversarial_ase:
            if self.rank == 0 or not self.is_distributed:
                print(f"Adversarial training enabled:")
                print(f"  - Erasure: threshold={self.ase_threshold}, weight={self.ase_loss_weight}")
                print(f"  - npe: epsilon={self.npe_epsilon}, weight={self.npe_loss_weight}")
                print(f"  - Clean loss: weight={self.clean_loss_weight}")
        # Optimize Adapters + Head
        model = self.model.module if hasattr(self.model, 'module') else self.model
        params = model.get_trainable_params()
        self.optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt.niter, eta_min=opt.lr * 0.01)
        
        self.best_avg_acc = 0.0
    
    def _get_base_model(self):
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.float().to(self.device)
            
            if self.use_adversarial_ase:
                loss = self._train_step_with_adversarial_ase(images, labels)
            else:
                loss = self._train_step_normal(images, labels)
            
            total_loss += loss.item()
            
            # Compute accuracy
            with torch.no_grad():
                logits = self.model(images)
                if logits.dim() > 1:
                    logits = logits.view(-1)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            if (step + 1) % self.opt.log_freq == 0:
                avg_loss = total_loss / (step + 1)
                acc = 100 * correct / total
                if self.rank == 0 or not self.is_distributed:
                    print(f'Epoch {epoch} Step {step+1}/{len(train_loader)}: Loss={avg_loss:.4f}, Acc={acc:.2f}%')
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        if self.rank == 0 or not self.is_distributed:
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        
        return train_loss, train_acc
    
    def _train_step_normal(self, images, labels):
        self.optimizer.zero_grad()
        logits = self.model(images)
        loss = self.criterion(logits.view(-1), labels)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _train_step_with_adversarial_ase(self, images, labels):
        """Adversarial training step: erasure + npe."""
        self.optimizer.zero_grad()
        
        logits_clean, attentions = self.model(images, output_attentions=True)
        loss_clean = self.criterion(logits_clean.view(-1), labels)
        
        loss_ased = self._compute_ase_loss(images, labels, attentions)
        loss_npe = self._compute_npe_loss(images, labels)
        
        loss_total = (self.clean_loss_weight * loss_clean + 
                      self.ase_loss_weight * loss_ased + 
                      self.npe_loss_weight * loss_npe)
        
        loss_total.backward()
        self.optimizer.step()
        
        return loss_total
    
    def _compute_ase_loss(self, images, labels, attentions):
        """Compute erasure loss on fake images only."""
        fake_mask = labels == 1
        
        if not fake_mask.any():
            return torch.tensor(0.0, device=images.device)
        
        fake_images = images[fake_mask]
        fake_labels = labels[fake_mask]
        
        fake_attentions = tuple(attn[fake_mask] for attn in attentions)
        
        with torch.no_grad():
            model_base = self._get_base_model()
            n_prefix_tokens = model_base.n_prefix_tokens
            mask = generate_erasure_mask(
                fake_attentions, 
                image_size=fake_images.shape[-2:],
                threshold=self.ase_threshold,
                n_prefix_tokens=n_prefix_tokens
            )
            fake_images_ased = fake_images * mask
        
        logits_ased = self.model(fake_images_ased, output_attentions=False)
        loss_ased = self.criterion(logits_ased.view(-1), fake_labels)
        
        return loss_ased
    
    def _compute_npe_loss(self, images, labels):
        """Compute npe adversarial loss on fake images only."""
        fake_mask = labels == 1
        
        if not fake_mask.any():
            return torch.tensor(0.0, device=images.device)
        
        fake_images = images[fake_mask]
        fake_labels = labels[fake_mask]
        fake_images_adv = self._generate_npe_samples(fake_images, fake_labels)
        
        logits_adv = self.model(fake_images_adv, output_attentions=False)
        loss_npe = self.criterion(logits_adv.view(-1), fake_labels)
        
        return loss_npe
    
    def _generate_npe_samples(self, images, labels):
        images_for_grad = images.clone().detach().requires_grad_(True)
        
        self.model.eval()
        logits = self.model(images_for_grad, output_attentions=False)
        loss = self.criterion(logits.view(-1), labels)
        
        data_grad = torch.autograd.grad(loss, images_for_grad, 
                                         retain_graph=False, 
                                         create_graph=False)[0]
        
        perturbed_images = npe_attack(images, self.npe_epsilon, data_grad)
        
        self.model.train()
        
        return perturbed_images.detach()
    
    def evaluate(self):
        if self.is_distributed:
            eval_model = self._get_base_model()
        else:
            eval_model = self.model
        
        eval_model.eval()
        results = {}
        
        for val_name in VALS:
            test_path = os.path.join(self.opt.dataroot, 'test', val_name)
            if not os.path.exists(test_path):
                continue
            
            self.opt.isTrain = False
            orig_sample_list = self.opt.sample_list
            self.opt.sample_list = None
            test_dataset = dataset_folder(self.opt, test_path)
            self.opt.sample_list = orig_sample_list
            self.opt.isTrain = True
            
            # Distributed: use DistributedSampler
            if self.is_distributed:
                from torch.utils.data.distributed import DistributedSampler
                test_sampler = DistributedSampler(
                    test_dataset,
                    num_replicas=self.opt.world_size,
                    rank=self.rank,
                    shuffle=False
                )
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=self.opt.batch_size, 
                    shuffle=False,
                    sampler=test_sampler,
                    num_workers=self.opt.num_threads, 
                    pin_memory=True,
                    prefetch_factor=2, 
                    persistent_workers=True if self.opt.num_threads > 0 else False
                )
            else:
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=self.opt.batch_size, 
                    shuffle=False,
                    num_workers=self.opt.num_threads, 
                    pin_memory=True,
                    prefetch_factor=2, 
                    persistent_workers=True if self.opt.num_threads > 0 else False
                )
            
            all_probs, all_labels = [], []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    logits = eval_model(images)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(labels.numpy().tolist())
            
            # Distributed: gather predictions from all GPUs
            if self.is_distributed:
                import torch.distributed as dist
                all_probs_gathered = [None] * self.opt.world_size
                all_labels_gathered = [None] * self.opt.world_size
                dist.all_gather_object(all_probs_gathered, all_probs)
                dist.all_gather_object(all_labels_gathered, all_labels)
                
                all_probs = []
                all_labels = []
                for probs, labels in zip(all_probs_gathered, all_labels_gathered):
                    all_probs.extend(probs)
                    all_labels.extend(labels)
            
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            all_preds = (all_probs > 0.5).astype(int)
            
            acc = accuracy_score(all_labels, all_preds) * 100
            if len(np.unique(all_labels)) > 1:
                ap = average_precision_score(all_labels, all_probs) * 100
            else:
                ap = acc
            
            real_mask = all_labels == 0
            fake_mask = all_labels == 1
            r_acc = (all_preds[real_mask] == 0).sum() / real_mask.sum() * 100 if real_mask.sum() > 0 else 0.0
            f_acc = (all_preds[fake_mask] == 1).sum() / fake_mask.sum() * 100 if fake_mask.sum() > 0 else 0.0
            
            results[val_name] = {'acc': acc, 'ap': ap, 'r_acc': r_acc, 'f_acc': f_acc}
            
            if self.rank == 0 or not self.is_distributed:
                print(f'{val_name:15s}: Acc={acc:6.2f}%, AP={ap:6.2f}%, R_Acc={r_acc:6.2f}%, F_Acc={f_acc:6.2f}%')
        
        # Print summary
        if self.rank == 0 or not self.is_distributed:
            if results:
                avg_acc = np.mean([r['acc'] for r in results.values()])
                avg_ap = np.mean([r['ap'] for r in results.values()])
                avg_r_acc = np.mean([r['r_acc'] for r in results.values()])
                avg_f_acc = np.mean([r['f_acc'] for r in results.values()])
                print(f'{"Average":15s}: Acc={avg_acc:6.2f}%, AP={avg_ap:6.2f}%, R_Acc={avg_r_acc:6.2f}%, F_Acc={avg_f_acc:6.2f}%')
        
        return results
    
    def save_model(self, epoch, is_best=False):
        if self.is_distributed and self.rank != 0:
            return
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        lora_state_dict = {k: v for k, v in model.state_dict().items() 
                           if 'lora_' in k or 'head' in k}
        
        if is_best:
            torch.save({
                'epoch': epoch,
                'lora_state_dict': lora_state_dict,
                'best_acc': self.best_avg_acc,
            }, os.path.join(self.save_dir, 'best_model.pth'))
            print(f'Saved best model with avg_acc={self.best_avg_acc:.2f}%')
        else:
            torch.save({
                'epoch': epoch,
                'lora_state_dict': lora_state_dict,
            }, os.path.join(self.save_dir, f'epoch_{epoch}.pth'))
    
    def step_scheduler(self):
        self.scheduler.step()
    
    def update_best(self, avg_acc, epoch):
        if avg_acc > self.best_avg_acc:
            self.best_avg_acc = avg_acc
            self.save_model(epoch, is_best=True)
            return True
        return False
