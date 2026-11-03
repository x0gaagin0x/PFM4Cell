"""
Encoder
"""
import os
import timm
import torch
from torch import nn
from conch.load_conch import create_model_from_pretrained

# HIPT256
from HIPT.hipt_model_utils import get_vit256

# MUSK
from timm.models import create_model
from musk import utils, modeling
from transformers import XLMRobertaTokenizer

# CONCH
conch_model_cfg = {
    "embed_dim": 512,
    "embed_dim_caption": 768,
    "vision_cfg": {
        "image_size": 448,
        "patch_size": 16,
        "attentional_pool_caption": True,
        "attentional_pool_contrast": True,
        "attn_pooler_heads": 8,
        "n_queries_contrast": 1,
        "n_queries_caption": 256,
        "output_tokens": True
    },
    "text_cfg": {
        "context_length": 128,
        "vocab_size": 32007,
        "width": 768,
        "heads": 12,
        "layers": 12,
        "embed_cls": True,
        "output_tokens": True
    },
    "multimodal_cfg": {
        "context_length": 128,
        "vocab_size": 32007,
        "width": 768,
        "heads": 12,
        "layers": 12
    },
    "custom_text": True
}

## Function for loading UNI series model
def load_UNI(model_dir, device):
    model = timm.create_model( "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, 
                              num_classes=0, dynamic_img_size=True)
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device), strict=True)
    model.eval()
    
    return model

def load_UNI2h(model_dir, device):
    timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
    
    model = timm.create_model(pretrained=False, **timm_kwargs)
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device), strict=True)
    model.eval()
    
    return model

def load_Virchow(model_dir, device):
    timm_kwargs = {
            'model_name': 'vit_huge_patch14_224',
            'img_size': 224,
            'patch_size': 14,
            'depth': 32,
            'num_heads': 16,
            'init_values': 1e-5,
            'embed_dim': 1280,
            'mlp_ratio': 5.3375,
            'num_classes': 0,
            'no_embed_class': False,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'dynamic_img_size': True
        }
    
    model = timm.create_model(pretrained=False, **timm_kwargs)
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device), strict=True)
    model.eval()
    
    return model

def load_Virchow2(model_dir, device):
    timm_kwargs = {
            'model_name': 'vit_huge_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 32,
            'num_heads': 16,
            'init_values': 1e-5, 
            'embed_dim': 1280,
            'mlp_ratio': 5.3375,
            'num_classes': 0, 
            'no_embed_class': False,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 4,
            'dynamic_img_size': True
        }
    
    model = timm.create_model(pretrained=False, **timm_kwargs)
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device), strict=True)
    model.eval()
    
    return model

def load_gigapath(model_dir, device):
    timm_kwargs = {
    'model_name': 'vit_giant_patch14_dinov2', 
    'num_classes': 0,
    'global_pool': 'token',
    'img_size': 224,
    'in_chans': 3,
    'patch_size': 16, 
    'embed_dim': 1536,
    'depth': 40,
    'num_heads': 24,
    'init_values': 1e-05,
    'mlp_ratio': 5.33334
}
    
    model = timm.create_model(pretrained=False, **timm_kwargs)
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device), strict=True)
    model.eval()
    
    return model



def load_musk(model_dir):
    model_config = "musk_large_patch16_384"
    model = create_model(model_config).eval()
    utils.load_model_and_may_interpolate(os.path.join(model_dir, 'model.safetensors'), model, 'model|module', '')
    
    return model

class Encoder(nn.Module):
    def __init__(self, enc_type, model_dir=None):
        super().__init__()
        
        self.enc_type = enc_type
            
        if enc_type == 'UNI2-h':
            self.model = load_UNI2h(model_dir=model_dir, device='cpu')
        
        elif enc_type == 'UNI':
            self.model = load_UNI(model_dir=model_dir, device='cpu')
 
        elif enc_type == 'Virchow2':
            self.model = load_Virchow2(model_dir=model_dir, device='cpu')
        
        elif enc_type == 'Virchow':
            self.model = load_Virchow(model_dir=model_dir, device='cpu')
        
        elif enc_type == 'CONCH':
            self.model, self.preprocess = create_model_from_pretrained(conch_model_cfg,
                                                                       os.path.join(model_dir, "pytorch_model.bin"),
                                                                       force_image_size=224)
        elif enc_type == 'MUSK':
            self.model = load_musk(model_dir)
        
        elif enc_type == 'HIPT':
            self.model = get_vit256(pretrained_weights=model_dir)
        
        elif enc_type == 'gigapath':
            self.model = load_gigapath(model_dir=model_dir, device='cpu')
        
        else:
            raise ValueError(
            f"No supported encoder: enc_type={enc_type}, model_dir={model_dir}, no enc"
            )
        
    def forward(self, x):
        if self.enc_type == 'UNI2-h':
            emb = self.model.forward_features(x)
            cls_emb = emb[:, 0]
            patch_emb = emb[:, 9:]
        
        elif self.enc_type == 'UNI':
            emb = self.model.forward_features(x)
            cls_emb = emb[:, 0]
            patch_emb = emb[:, 1:]
            
        elif self.enc_type == 'Virchow2':
            emb = self.model.forward_features(x)
            cls_emb = emb[:, 0]
            patch_emb = emb[:, 5:]
        
        elif self.enc_type == 'Virchow':
            emb = self.model.forward_features(x)
            cls_emb = emb[:, 0]
            patch_emb = emb[:, 1:]
        
        elif self.enc_type == 'CONCH':
            cls_emb, patch_emb = self.model._encode_image(x, normalize=False)
        
        elif self.enc_type == 'MUSK':
            emb = self.model(image=x, with_head=False, out_norm=False, ms_aug=False, return_global=False)[0]
            cls_emb = emb[:, 0]
            patch_emb = emb[:, 1:]
        
        elif self.enc_type == 'HIPT':
            emb = self.model(x)
            cls_emb = emb[:, 0]
            patch_emb = emb[:, 1:]
        
        elif self.enc_type == 'gigapath':
            emb = self.model.forward_features(x)
            cls_emb = emb[:, 0]
            patch_emb = emb[:, 1:]
        
        return cls_emb, patch_emb