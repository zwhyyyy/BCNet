import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

dinov3_path = os.path.join(os.path.dirname(__file__), 'dinov3')
if dinov3_path not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from dinov3.hub.backbones import dinov3_vith16plus
try:
    from safetensors.torch import load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available, falling back to torch.load")

LOCAL_WEIGHT_DIR = 'your/path' # TODO dinov3 840M Folder
LOCAL_WEIGHT_PATH = os.path.join(LOCAL_WEIGHT_DIR, 'model.safetensors')

def _is_hf_style_state_dict(state_dict: dict) -> bool:
    # HuggingFace-style keys typically include "embeddings." and/or "layer.N."
    for k in state_dict.keys():
        if k.startswith("embeddings.") or k.startswith("layer."):
            return True
    return False


def _convert_hf_to_dinov3_state_dict(hf_sd: dict, backbone: nn.Module) -> dict:
    out = {}

    # ----- embeddings -----
    if "embeddings.cls_token" in hf_sd:
        out["cls_token"] = hf_sd["embeddings.cls_token"]
    if "embeddings.mask_token" in hf_sd:
        # HF: [1, 1, C], dinov3: [1, C]
        mt = hf_sd["embeddings.mask_token"]
        if mt.ndim == 3 and mt.shape[0] == 1 and mt.shape[1] == 1:
            mt = mt.squeeze(1)
        out["mask_token"] = mt
    # dinov3 uses "storage_tokens" for registers
    if "embeddings.register_tokens" in hf_sd:
        out["storage_tokens"] = hf_sd["embeddings.register_tokens"]
    # patch embed
    if "embeddings.patch_embeddings.weight" in hf_sd:
        out["patch_embed.proj.weight"] = hf_sd["embeddings.patch_embeddings.weight"]
    if "embeddings.patch_embeddings.bias" in hf_sd:
        out["patch_embed.proj.bias"] = hf_sd["embeddings.patch_embeddings.bias"]

    # Determine number of blocks from backbone
    n_blocks = getattr(backbone, "n_blocks", None)
    if n_blocks is None and hasattr(backbone, "blocks"):
        n_blocks = len(backbone.blocks)
    if n_blocks is None:
        # fallback: infer from keys
        n_blocks = 0
        for k in hf_sd.keys():
            if k.startswith("layer."):
                try:
                    idx = int(k.split(".")[1])
                    n_blocks = max(n_blocks, idx + 1)
                except Exception:
                    pass

    # ----- per-layer mapping -----
    for i in range(int(n_blocks)):
        # norms
        for name in ("weight", "bias"):
            k1 = f"layer.{i}.norm1.{name}"
            k2 = f"layer.{i}.norm2.{name}"
            if k1 in hf_sd:
                out[f"blocks.{i}.norm1.{name}"] = hf_sd[k1]
            if k2 in hf_sd:
                out[f"blocks.{i}.norm2.{name}"] = hf_sd[k2]

        # layer scales
        k_ls1 = f"layer.{i}.layer_scale1.lambda1"
        k_ls2 = f"layer.{i}.layer_scale2.lambda1"
        if k_ls1 in hf_sd:
            out[f"blocks.{i}.ls1.gamma"] = hf_sd[k_ls1]
        if k_ls2 in hf_sd:
            out[f"blocks.{i}.ls2.gamma"] = hf_sd[k_ls2]

        # attention output projection
        for name in ("weight", "bias"):
            k_o = f"layer.{i}.attention.o_proj.{name}"
            if k_o in hf_sd:
                out[f"blocks.{i}.attn.proj.{name}"] = hf_sd[k_o]

        # QKV: concatenate q,k,v weights/biases into qkv
        q_w = hf_sd.get(f"layer.{i}.attention.q_proj.weight", None)
        k_w = hf_sd.get(f"layer.{i}.attention.k_proj.weight", None)
        v_w = hf_sd.get(f"layer.{i}.attention.v_proj.weight", None)
        if q_w is not None and k_w is not None and v_w is not None:
            out[f"blocks.{i}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = hf_sd.get(f"layer.{i}.attention.q_proj.bias", None)
        k_b = hf_sd.get(f"layer.{i}.attention.k_proj.bias", None)
        v_b = hf_sd.get(f"layer.{i}.attention.v_proj.bias", None)
        if q_b is not None and k_b is not None and v_b is not None:
            out[f"blocks.{i}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        for name in ("weight", "bias"):
            g = hf_sd.get(f"layer.{i}.mlp.gate_proj.{name}", None)
            u = hf_sd.get(f"layer.{i}.mlp.up_proj.{name}", None)
            d = hf_sd.get(f"layer.{i}.mlp.down_proj.{name}", None)
            if g is not None:
                out[f"blocks.{i}.mlp.w1.{name}"] = g
            if u is not None:
                out[f"blocks.{i}.mlp.w2.{name}"] = u
            if d is not None:
                out[f"blocks.{i}.mlp.w3.{name}"] = d

    # final norms
    # dinov3 may have norm, cls_norm, local_cls_norm depending on untie flags; HF weights typically have "norm.*"
    if "norm.weight" in hf_sd and hasattr(backbone, "norm"):
        out["norm.weight"] = hf_sd["norm.weight"]
    if "norm.bias" in hf_sd and hasattr(backbone, "norm"):
        out["norm.bias"] = hf_sd["norm.bias"]
    if "cls_norm.weight" in hf_sd and hasattr(backbone, "cls_norm"):
        out["cls_norm.weight"] = hf_sd["cls_norm.weight"]
    if "cls_norm.bias" in hf_sd and hasattr(backbone, "cls_norm"):
        out["cls_norm.bias"] = hf_sd["cls_norm.bias"]
    if "local_cls_norm.weight" in hf_sd and hasattr(backbone, "local_cls_norm"):
        out["local_cls_norm.weight"] = hf_sd["local_cls_norm.weight"]
    if "local_cls_norm.bias" in hf_sd and hasattr(backbone, "local_cls_norm"):
        out["local_cls_norm.bias"] = hf_sd["local_cls_norm.bias"]

    return out

def _align_and_filter_state_dict_for_backbone(backbone: nn.Module, sd: dict) -> dict:
    target = backbone.state_dict()
    out = {}
    for k, v in sd.items():
        if k not in target:
            continue
        tv = target[k]
        vv = v

        # If shapes match, keep
        if vv.shape == tv.shape:
            out[k] = vv
            continue

        # Common case: singleton dimension
        if vv.ndim == tv.ndim + 1 and vv.shape[0] == 1 and vv.shape[1:] == tv.shape:
            vv2 = vv.squeeze(0)
            if vv2.shape == tv.shape:
                out[k] = vv2
                continue
        if vv.ndim == tv.ndim + 1 and vv.shape[-1] == 1 and vv.shape[:-1] == tv.shape:
            vv2 = vv.squeeze(-1)
            if vv2.shape == tv.shape:
                out[k] = vv2
                continue
        if vv.ndim == 3 and tv.ndim == 2 and vv.shape[0] == 1 and vv.shape[1:] == tv.shape:
            vv2 = vv.squeeze(0)
            if vv2.shape == tv.shape:
                out[k] = vv2
                continue
        if vv.ndim == 3 and tv.ndim == 2 and vv.shape[1] == 1 and (vv.shape[0], vv.shape[2]) == tv.shape:
            vv2 = vv.squeeze(1)
            if vv2.shape == tv.shape:
                out[k] = vv2
                continue

        # Common case: Linear weight transposed
        if vv.ndim == 2 and tv.ndim == 2 and vv.t().shape == tv.shape:
            out[k] = vv.t()
            continue

        # Otherwise: skip
        # (we intentionally do not throw; better to load what matches)
    return out


def _make_fully_matched_backbone_state_dict(backbone: nn.Module, mapped_aligned: dict) -> dict:
    full = backbone.state_dict()
    full.update(mapped_aligned)
    return full


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation): W' = W + BA, where B: [out, r], A: [r, in]."""
    def __init__(self, original_layer: nn.Linear, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return original_out + lora_out


class RobustDeepFakeHead(nn.Module):
    def __init__(self, embed_dim, num_classes=1, n_prefix_tokens=5, topk=16):
        super().__init__()
        self.n_prefix_tokens = n_prefix_tokens
        self.topk = topk
        self.embed_dim = embed_dim
        
        self.proj = nn.Linear(embed_dim, 512)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512 * 3, num_classes)
    
    def forward(self, x):
        """x: [B, L, C]"""
        x_proj = self.dropout(self.act(self.proj(x)))
        
        cls_token = x_proj[:, 0]
        patch_tokens = x_proj[:, self.n_prefix_tokens:]
        
        avg_pool = patch_tokens.mean(dim=1)
        
        k = min(self.topk, patch_tokens.shape[1])
        topk_vals, _ = patch_tokens.topk(k, dim=1)
        topk_pool = topk_vals.mean(dim=1)
        
        combined = torch.cat([cls_token, avg_pool, topk_pool], dim=1)
        combined = F.normalize(combined, dim=-1)
        
        return self.fc(combined)


class DINOv3Classifier(nn.Module):
    """DINOv3 ViT-H+/16 + LoRA fine-tuning with adversarial erasure support."""
    def __init__(self, num_classes=1, lora_r=16, lora_alpha=32, lora_dropout=0.1, 
                 lora_layers=None):
        super().__init__()
        
        if lora_layers is None:
            lora_layers = list(range(32))
        self.lora_layers = lora_layers
        
        self.backbone = dinov3_vith16plus(pretrained=False)
        
        if os.path.exists(LOCAL_WEIGHT_PATH):
            print(f"Loading DINOv3 ViT-H+/16 weights from {LOCAL_WEIGHT_PATH}")
            if SAFETENSORS_AVAILABLE:
                state_dict = safetensors_load(LOCAL_WEIGHT_PATH)

                clean_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        clean_state_dict[k[9:]] = v
                    elif k.startswith('model.'):
                        clean_state_dict[k[6:]] = v
                    else:
                        clean_state_dict[k] = v

                # If HF-style, do key mapping/alignment
                if _is_hf_style_state_dict(clean_state_dict):
                    mapped = _convert_hf_to_dinov3_state_dict(clean_state_dict, self.backbone)
                    mapped = _align_and_filter_state_dict_for_backbone(self.backbone, mapped)
                    full = _make_fully_matched_backbone_state_dict(self.backbone, mapped)
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(full, strict=True)
                    print(f"[BackboneLoad] HF->dinov3 mapped+aligned keys: {len(mapped)} (strict=True fully matched)")
                else:
                    aligned = _align_and_filter_state_dict_for_backbone(self.backbone, clean_state_dict)
                    full = _make_fully_matched_backbone_state_dict(self.backbone, aligned)
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(full, strict=True)

                # strict=True should make these empty; keep prints as assertions
                if missing_keys:
                    print(f"Warning: Missing keys (unexpected!): {missing_keys[:10]}... (showing first 10)")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys (unexpected!): {unexpected_keys[:10]}... (showing first 10)")
                print("Successfully loaded backbone weights from safetensors")
            else:
                # Fallback to torch.load if safetensors not available
                state_dict = torch.load(LOCAL_WEIGHT_PATH, map_location='cpu', weights_only=False)
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                clean_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        clean_state_dict[k[9:]] = v
                    elif k.startswith('model.'):
                        clean_state_dict[k[6:]] = v
                    else:
                        clean_state_dict[k] = v
                if _is_hf_style_state_dict(clean_state_dict):
                    mapped = _convert_hf_to_dinov3_state_dict(clean_state_dict, self.backbone)
                    mapped = _align_and_filter_state_dict_for_backbone(self.backbone, mapped)
                    full = _make_fully_matched_backbone_state_dict(self.backbone, mapped)
                    self.backbone.load_state_dict(full, strict=True)
                    print(f"[BackboneLoad] HF->dinov3 mapped+aligned keys: {len(mapped)} (strict=True fully matched)")
                else:
                    aligned = _align_and_filter_state_dict_for_backbone(self.backbone, clean_state_dict)
                    full = _make_fully_matched_backbone_state_dict(self.backbone, aligned)
                    self.backbone.load_state_dict(full, strict=True)
                print("Loaded backbone weights using torch.load (fallback, strict=True fully matched)")
        else:
            # Fallback to hub if local weights not found
            print(f"Local weights not found at {LOCAL_WEIGHT_PATH}, loading from hub...")
            self.backbone = dinov3_vith16plus(pretrained=True)
            print("Loaded DINOv3 ViT-H+/16 weights from hub")
        # Config
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        self.n_storage_tokens = self.backbone.n_storage_tokens
        self.n_prefix_tokens = 1 + self.n_storage_tokens
        self.num_heads = self.backbone.num_heads
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self._inject_lora(lora_r, lora_alpha, lora_dropout)
        
        self.head = RobustDeepFakeHead(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            n_prefix_tokens=self.n_prefix_tokens,
            topk=32
        )
        # Stats
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Injected LoRA at layers: {lora_layers}, r={lora_r}, alpha={lora_alpha}")
        print(f"Total trainable params: {trainable_params / 1e6:.2f}M")
    
    def _inject_lora(self, r, alpha, dropout):
        for i in self.lora_layers:
            block = self.backbone.blocks[i]
            original_qkv = block.attn.qkv
            block.attn.qkv = LoRALayer(original_qkv, r=r, lora_alpha=alpha, lora_dropout=dropout)
    
    def forward_features(self, x, output_attentions=False):
        """Extract features. Returns (features, attentions) if output_attentions=True."""
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x, masks=None)
        self._current_H = H
        self._current_W = W
        
        attentions = [] if output_attentions else None
        
        for i, blk in enumerate(self.backbone.blocks):
            rope_sincos = self.backbone.rope_embed(H=H, W=W) if self.backbone.rope_embed else None
            
            if output_attentions:
                attn_out, attn_weights = self._forward_block_with_attn(blk, x, rope_sincos)
                x = attn_out
                attentions.append(attn_weights)
            else:
                x = blk(x, rope_sincos)
        
        # Final LayerNorm
        # Final LayerNorm
        if self.backbone.untie_cls_and_patch_norms:
            x_cls = self.backbone.cls_norm(x[:, :self.n_storage_tokens + 1])
            x_patch = self.backbone.norm(x[:, self.n_storage_tokens + 1:])
            features = torch.cat([x_cls, x_patch], dim=1)
        else:
            features = self.backbone.norm(x)
        
        if output_attentions:
            return features, tuple(attentions)
        return features
    
    def _forward_block_with_attn(self, blk, x, rope_sincos):
        attn_input = blk.norm1(x)
        attn_out, attn_weights = self._attention_with_weights(blk.attn, attn_input, rope_sincos)
        attn_out = blk.ls1(attn_out)
        x = x + attn_out
        
        # MLP with residual
        mlp_out = blk.mlp(blk.norm2(x))
        mlp_out = blk.ls2(mlp_out)
        x = x + mlp_out
        
        return x, attn_weights
    
    def _attention_with_weights(self, attn_module, x, rope):
        B, N, C = x.shape
        qkv = attn_module.qkv(x)
        qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        
        if rope is not None:
            q, k = attn_module.apply_rope(q, k, rope)
        
        # Manually compute attention weights
        # Manually compute attention weights
        scale = (C // attn_module.num_heads) ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = attn_module.proj(out)
        out = attn_module.proj_drop(out)
        
        return out, attn_weights
    
    def forward(self, x, output_attentions=False):
        if output_attentions:
            features, attentions = self.forward_features(x, output_attentions=True)
            logits = self.head(features)
            return logits, attentions
        else:
            features = self.forward_features(x)
            return self.head(features)
    
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    
    def get_trainable_params(self):
        """Get trainable parameters (LoRA + head)."""
        params = []
        for i in self.lora_layers:
            lora_layer = self.backbone.blocks[i].attn.qkv
            # 检查是否是LoRALayer类型
            if hasattr(lora_layer, 'lora_A') and hasattr(lora_layer, 'lora_B'):
                params.extend([lora_layer.lora_A.weight, lora_layer.lora_B.weight])
            else:
                raise ValueError(f"Layer {i} is not a LoRALayer. Make sure LoRA is properly injected.")
        params.extend(self.head.parameters())
        return params


def parse_lora_layers(lora_layers_str):
    if lora_layers_str == 'all':
        return list(range(32))
    elif lora_layers_str == 'last4':
        return [28, 29, 30, 31]
    elif lora_layers_str == 'last8':
        return list(range(24, 32))
    elif lora_layers_str == 'last12':
        return list(range(20, 32))
    elif lora_layers_str == 'last16':
        return list(range(16, 32))
    else:
        return [int(x.strip()) for x in lora_layers_str.split(',')]


def generate_erasure_mask(attentions, image_size=(224, 224), threshold=0.6, n_prefix_tokens=5):
    """Generate erasure mask from attention weights. Returns [B, 1, H, W] binary mask."""
    last_attn = attentions[-1]
    
    # CLS attention to patch tokens
    cls_attn = last_attn[:, :, 0, n_prefix_tokens:]
    cls_attn = cls_attn.mean(dim=1)
    
    B, N = cls_attn.shape
    min_val = cls_attn.min(dim=1, keepdim=True)[0]
    max_val = cls_attn.max(dim=1, keepdim=True)[0]
    cls_attn = (cls_attn - min_val) / (max_val - min_val + 1e-8)
    
    grid_size = int(N ** 0.5)
    if grid_size * grid_size != N:
        grid_size = int(np.sqrt(N))
        if grid_size * grid_size != N:
            raise ValueError(f"Cannot reshape attention map: N={N} is not a perfect square")
    attn_map = cls_attn.view(B, 1, grid_size, grid_size)
    attn_map = F.interpolate(attn_map, size=image_size, mode='bilinear', align_corners=False)
    mask = (attn_map < threshold).float()
    
    return mask.detach()


def create_model(opt):
    lora_layers_str = getattr(opt, 'lora_layers', 'all')
    lora_layers = parse_lora_layers(lora_layers_str)
    lora_r = getattr(opt, 'lora_r', 16)
    lora_alpha = getattr(opt, 'lora_alpha', 32)
    lora_dropout = getattr(opt, 'lora_dropout', 0.1)
    
    model = DINOv3Classifier(
        num_classes=1,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_layers=lora_layers
    )
    
    # For multi-GPU, device assignment is handled by DDP
    if not hasattr(opt, 'distributed') or not opt.distributed:
        if opt.gpu_id >= 0 and torch.cuda.is_available():
            model = model.cuda(opt.gpu_id)
    
    return model
