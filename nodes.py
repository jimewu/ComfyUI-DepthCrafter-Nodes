import os
import torch
import math
import gc
import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

from .depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from .depthcrafter.depth_crafter_ppl import DepthCrafterPipeline


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class DepthCrafterNode:
    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self, *args, **kwargs):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None
        
    CATEGORY = "DepthCrafter"


class DownloadAndLoadDepthCrafterModel(DepthCrafterNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "enable_model_cpu_offload": ("BOOLEAN", {"default": True}),
            "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("DEPTHCRAFTER_MODEL",)
    RETURN_NAMES = ("depthcrafter_model",)
    FUNCTION = "load_model"
    DESCRIPTION = """
下載並載入 DepthCrafter 模型 (V7)

【AMD AI MAX+ 395 建議設定】
• enable_model_cpu_offload: ✅ 開啟
• enable_sequential_cpu_offload: ❌ 關閉
"""

    def load_model(self, enable_model_cpu_offload, enable_sequential_cpu_offload):
        device = mm.get_torch_device()

        model_dir = os.path.join(folder_paths.models_dir, "depthcrafter")
        os.makedirs(model_dir, exist_ok=True)

        unet_path = os.path.join(model_dir, "tencent_DepthCrafter")
        pretrain_path = os.path.join(model_dir, "stabilityai_stable-video-diffusion-img2vid-xt")

        depthcrafter_files_to_download = [
            "config.json",
            "diffusion_pytorch_model.safetensors",
        ]
        svd_files_to_download = [
            "feature_extractor/preprocessor_config.json",
            "image_encoder/config.json",
            "image_encoder/model.fp16.safetensors",
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "model_index.json",
        ]

        self.start_progress(len(svd_files_to_download) + len(depthcrafter_files_to_download))

        from huggingface_hub import hf_hub_download

        if not os.path.exists(unet_path):
            print(f"Downloading UNet model to: {unet_path}")
            for path in depthcrafter_files_to_download:
                hf_hub_download(
                    repo_id="tencent/DepthCrafter",
                    filename=path,
                    local_dir=unet_path,
                    local_dir_use_symlinks=False,
                    revision="c1a22b53f8abf80cd0b025adf29e637773229eca",
                )
                self.update_progress()

        if not os.path.exists(pretrain_path):
            print(f"Downloading pre-trained pipeline to: {pretrain_path}")
            for path in svd_files_to_download:
                hf_hub_download(
                    repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
                    filename=path,
                    local_dir=pretrain_path,
                    local_dir_use_symlinks=False,
                    revision="9e43909513c6714f1bc78bcb44d96e733cd242aa",
                )
                self.update_progress()

        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        pipe = DepthCrafterPipeline.from_pretrained(
            pretrain_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
        )
        
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        elif enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

        depthcrafter_model = {
            "pipe": pipe,
            "device": device,
        }

        self.end_progress()

        return (depthcrafter_model,)


class DepthCrafter(DepthCrafterNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depthcrafter_model": ("DEPTHCRAFTER_MODEL", ),
                "images": ("IMAGE", ),
                "force_size": ("BOOLEAN", {"default": True}),
                "num_inference_steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "auto_optimize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "自動計算最佳配置。強烈建議保持開啟。"
                }),
                "prefer_fewer_tiles": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "優先減少 tile 數量。"
                }),
                "max_memory_usage_gb": ("FLOAT", {
                    "default": 110.0,
                    "min": 40.0,
                    "max": 128.0,
                    "step": 5.0,
                    "tooltip": "最大可用記憶體 (GB)。"
                }),
                "window_size_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "手動設定 window_size。0 = 自動。"
                }),
                "spatial_tile_size_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1536,
                    "step": 64,
                    "tooltip": "手動設定 tile_size。0 = 自動。"
                }),
                "vae_encode_chunk_size": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 4, 
                    "step": 1,
                }),
                "vae_decode_chunk_size": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 4, 
                    "step": 1,
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_maps",)
    FUNCTION = "process"
    DESCRIPTION = """
DepthCrafter V7 - AMD AI MAX+ 395 優化版本

【關鍵修復】
• 強制 VAE 使用 float32 進行編碼/解碼
• 修復維度處理錯誤
• 更穩定的數值處理

【建議設定】
• auto_optimize: ✅ 開啟
• max_memory_usage_gb: 110
• vae_encode/decode_chunk_size: 1
"""
    
    def process(
        self, 
        depthcrafter_model, 
        images, 
        force_size, 
        num_inference_steps, 
        guidance_scale, 
        auto_optimize=True,
        prefer_fewer_tiles=True,
        max_memory_usage_gb=110.0,
        window_size_override=0,
        spatial_tile_size_override=0,
        vae_encode_chunk_size=1,
        vae_decode_chunk_size=1,
    ):
        aggressive_cleanup()
        
        device = depthcrafter_model['device']
        pipe = depthcrafter_model['pipe']
        
        B, H, W, C = images.shape

        if force_size:
            width = round(W / 64) * 64
            height = round(H / 64) * 64
            width = max(64, width)
            height = max(64, height)

            if width != W or height != H:
                print(f"DepthCrafter: Resizing input from {W}x{H} to {width}x{height}")
                images_for_resize = images.permute(0, 3, 1, 2)
                images_resized = torch.nn.functional.interpolate(
                    images_for_resize,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                )
                images = images_resized.permute(0, 2, 3, 1)
                H, W = height, width
            else:
                width = W
                height = H
        else:
            if W % 64 != 0 or H % 64 != 0:
                raise ValueError(
                    f"Input dimensions ({W}x{H}) not multiples of 64."
                )
            width = W
            height = H

        window_size = window_size_override if window_size_override > 0 else None
        spatial_tile_size = spatial_tile_size_override if spatial_tile_size_override > 0 else None
        
        # 使用 float32 輸入
        images = images.permute(0, 3, 1, 2)
        images = images.to(device=device, dtype=torch.float32)
        images = torch.clamp(images, 0, 1)
        
        self.start_progress(num_inference_steps * 10)
        
        aggressive_cleanup()
        
        with torch.inference_mode():
            result = pipe(
                images,
                height=height,
                width=width,
                output_type="pt",
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                window_size=window_size,
                overlap=None,
                track_time=False,
                progress_callback=self.update_progress,
                enable_spatial_tiling=True,
                spatial_tile_size=spatial_tile_size,
                spatial_tile_overlap=None,
                vae_encode_chunk_size=vae_encode_chunk_size,
                vae_decode_chunk_size=vae_decode_chunk_size,
                max_memory_usage_gb=max_memory_usage_gb,
                auto_optimize=auto_optimize,
                prefer_fewer_tiles=prefer_fewer_tiles,
            )
            
        res = result.frames[0]
        
        # 處理不同的輸出形狀
        print(f"[DepthCrafter] 輸出形狀: {res.shape}")
        
        # 確保是 [B, H, W, C] 或 [B, C, H, W] 格式
        if res.dim() == 5:
            # [1, B, C, H, W] -> [B, C, H, W]
            res = res.squeeze(0)
        
        if res.dim() == 4:
            if res.shape[1] == 3:
                # [B, C, H, W] -> [B, H, W, C]
                res = res.permute(0, 2, 3, 1)
        
        # 現在應該是 [B, H, W, C]
        print(f"[DepthCrafter] 處理後形狀: {res.shape}")
        
        # 修復無效值
        if torch.isnan(res).any() or torch.isinf(res).any():
            nan_count = torch.isnan(res).sum().item()
            inf_count = torch.isinf(res).sum().item()
            total = res.numel()
            print(f"[DepthCrafter] 修復無效值: NaN={nan_count}, Inf={inf_count} / {total}")
            res = torch.nan_to_num(res, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 轉換為灰階深度圖
        if res.dim() == 4 and res.shape[-1] == 3:
            # [B, H, W, 3] -> [B, H, W]
            res = res.mean(dim=-1)
        elif res.dim() == 4 and res.shape[-1] == 1:
            # [B, H, W, 1] -> [B, H, W]
            res = res.squeeze(-1)
        elif res.dim() == 3:
            # 已經是 [B, H, W]
            pass
        else:
            print(f"[DepthCrafter] 警告: 未預期的輸出形狀 {res.shape}")
        
        print(f"[DepthCrafter] 灰階形狀: {res.shape}")
        
        # 正規化
        res_min = res.min()
        res_max = res.max()
        print(f"[DepthCrafter] 深度範圍: min={res_min.item():.4f}, max={res_max.item():.4f}")
        
        if res_max - res_min > 1e-6:
            res = (res - res_min) / (res_max - res_min)
        else:
            print("[DepthCrafter] 警告: 深度圖變化很小，使用預設灰度")
            res = torch.ones_like(res) * 0.5
        
        # 確保是 3D [B, H, W]
        if res.dim() == 2:
            res = res.unsqueeze(0)
        
        # 擴展為 RGB [B, H, W, 3]
        depth_maps = res.unsqueeze(-1).expand(-1, -1, -1, 3)
        depth_maps = depth_maps.float().cpu().contiguous()
        
        print(f"[DepthCrafter] 最終輸出形狀: {depth_maps.shape}")
        
        self.end_progress()
        aggressive_cleanup()
        
        return (depth_maps,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadDepthCrafterModel": DownloadAndLoadDepthCrafterModel,
    "DepthCrafter": DepthCrafter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadDepthCrafterModel": "Download And Load DepthCrafter Model",
    "DepthCrafter": "DepthCrafter",
}
