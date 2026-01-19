import os
import torch
import math
import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

from .depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from .depthcrafter.depth_crafter_ppl import DepthCrafterPipeline


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
Downloads and loads the DepthCrafter model.
- enable_model_cpu_offload: If True, the model will be offloaded to the CPU. (Saves VRAM)
- enable_sequential_cpu_offload: If True, the model will be offloaded to the CPU in a sequential manner. (Saves the most VRAM but runs slowly)
Only enable one of the two at a time.
"""

    def load_model(self, enable_model_cpu_offload, enable_sequential_cpu_offload):
        device = mm.get_torch_device()

        model_dir = os.path.join(folder_paths.models_dir, "depthcrafter")
        os.makedirs(model_dir, exist_ok=True)

        # Paths to models
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

        # Check if models exist, if not download them
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

        # Load the custom UNet model
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # Load the pipeline
        pipe = DepthCrafterPipeline.from_pretrained(
            pretrain_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_local_files_only=True,
            low_cpu_mem_usage=True,
        )

        # Model setup
        pipe.enable_attention_slicing()
        
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
                "window_size": ("INT", {"default": 60, "min": 1, "max": 200}),
                "overlap": ("INT", {"default": 15, "min": 0, "max": 100}),
            },
            "optional": {
                # ==================== AMD/高解析度 Tiling 配置 ====================
                "enable_spatial_tiling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "啟用空間分塊處理，用於處理 2K+ 解析度影片"
                }),
                "spatial_tile_size": ("INT", {
                    "default": 1024, 
                    "min": 512, 
                    "max": 2048, 
                    "step": 64,
                    "tooltip": "每個空間 tile 的處理尺寸（像素）。建議：2K=1024, 4K=768。必須是 64 的倍數。"
                }),
                "spatial_tile_overlap": ("INT", {
                    "default": 192, 
                    "min": 64, 
                    "max": 512, 
                    "step": 32,
                    "tooltip": "空間 tiles 之間的重疊像素。較大的值可減少接縫，建議至少 128。"
                }),
                "vae_encode_chunk_size": ("INT", {
                    "default": 2, 
                    "min": 1, 
                    "max": 16, 
                    "step": 1,
                    "tooltip": "每次 VAE encoding 處理的幀數。128GB 記憶體建議 2-4。"
                }),
                "vae_decode_chunk_size": ("INT", {
                    "default": 4, 
                    "min": 1, 
                    "max": 16, 
                    "step": 1,
                    "tooltip": "每次 VAE decoding 處理的幀數。128GB 記憶體建議 4-8。"
                }),
                "max_memory_usage_gb": ("FLOAT", {
                    "default": 100.0,
                    "min": 16.0,
                    "max": 128.0,
                    "step": 8.0,
                    "tooltip": "最大允許的記憶體使用量 (GB)。128GB 系統建議設為 100-110。"
                }),
                # ==============================================================
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_maps",)
    FUNCTION = "process"
    DESCRIPTION = """
DepthCrafter - 影片深度圖生成
AMD gfx1151 + 128GB 統一記憶體優化版本

【基本參數】
• force_size: 自動調整輸入為 64 的倍數
• num_inference_steps: 去噪步數（更多=更好品質，更慢）
• guidance_scale: CFG 引導比例
• window_size: 時間滑動窗口大小（幀數）
• overlap: 時間窗口重疊（幀數）

【AMD/高解析度 Tiling 參數】
這些參數用於處理 2K+ 解析度的影片。

• enable_spatial_tiling: 啟用空間分塊（2K+ 必須啟用）
• spatial_tile_size: 每個 tile 的尺寸
  - 2K 影片: 1024 (建議)
  - 4K+ 影片: 768
• spatial_tile_overlap: tile 重疊像素（減少接縫）
  - 建議: 192
• vae_encode_chunk_size: VAE 編碼批次大小
  - 128GB 記憶體: 2-4
• vae_decode_chunk_size: VAE 解碼批次大小
  - 128GB 記憶體: 4-8
• max_memory_usage_gb: 最大記憶體使用量
  - 128GB 系統建議: 100-110

【記憶體建議】
• 如果 OOM: 降低 spatial_tile_size 或 chunk_size
• 如果有明顯接縫: 增加 spatial_tile_overlap
• 如果速度太慢: 增加 chunk_size（如記憶體允許）
"""
    
    def process(
        self, 
        depthcrafter_model, 
        images, 
        force_size, 
        num_inference_steps, 
        guidance_scale, 
        window_size, 
        overlap,
        # Optional AMD/Tiling parameters with defaults
        enable_spatial_tiling=True,
        spatial_tile_size=1024,
        spatial_tile_overlap=192,
        vae_encode_chunk_size=2,
        vae_decode_chunk_size=4,
        max_memory_usage_gb=100.0,
    ):
        device = depthcrafter_model['device']
        pipe = depthcrafter_model['pipe']
        
        B, H, W, C = images.shape
        orig_H, orig_W = H, W

        if force_size:
            # Round to nearest multiple of 64
            width = round(W / 64) * 64
            height = round(H / 64) * 64
            # Ensure minimum size is 64
            width = max(64, width)
            height = max(64, height)

            if width != W or height != H:
                print(f"DepthCrafter: Resizing input from {W}x{H} to {width}x{height} (multiples of 64)")
                # Permute for interpolation: B, H, W, C -> B, C, H, W
                images_for_resize = images.permute(0, 3, 1, 2)
                images_resized = torch.nn.functional.interpolate(
                    images_for_resize,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                )
                # Permute back: B, C, H, W -> B, H, W, C
                images = images_resized.permute(0, 2, 3, 1)
                H, W = height, width
            else:
                width = W
                height = H
        else:
            if W % 64 != 0 or H % 64 != 0:
                raise ValueError(
                    f"Input image dimensions ({W}x{H}) are not multiples of 64. "
                    f"Please resize your image to a multiple of 64 (e.g., {round(W / 64) * 64}x{round(H / 64) * 64}) "
                    f"or enable the 'force_size' option."
                )
            width = W
            height = H

        # 顯示 tiling 配置資訊
        max_dim = max(height, width)
        if enable_spatial_tiling and max_dim > spatial_tile_size:
            stride = spatial_tile_size - spatial_tile_overlap
            n_tiles_y = max(1, math.ceil((height - spatial_tile_overlap) / stride))
            n_tiles_x = max(1, math.ceil((width - spatial_tile_overlap) / stride))
            total_tiles = n_tiles_y * n_tiles_x
            print(f"DepthCrafter: Spatial Tiling enabled - {total_tiles} tiles ({n_tiles_x}x{n_tiles_y})")
            print(f"  - Tile size: {spatial_tile_size}, Overlap: {spatial_tile_overlap}")
            print(f"  - VAE encode chunk: {vae_encode_chunk_size}, VAE decode chunk: {vae_decode_chunk_size}")
            print(f"  - Max memory: {max_memory_usage_gb} GB")
        else:
            print(f"DepthCrafter: Processing at {width}x{height} (no spatial tiling needed)")

        # Permute images to [t, c, h, w] for the pipeline
        images = images.permute(0, 3, 1, 2)  # [B, C, H, W]
        images = images.to(device=device, dtype=torch.float16)
        images = torch.clamp(images, 0, 1)
        
        # Calculate total num of steps for progress bar
        if enable_spatial_tiling and max_dim > spatial_tile_size:
            stride = spatial_tile_size - spatial_tile_overlap
            n_tiles_y = max(1, math.ceil((height - spatial_tile_overlap) / stride))
            n_tiles_x = max(1, math.ceil((width - spatial_tile_overlap) / stride))
            total_tiles = n_tiles_y * n_tiles_x
            num_windows = math.ceil((B - window_size) / max(1, window_size - overlap)) + 1 if B > window_size else 1
            total_steps = num_inference_steps * num_windows * total_tiles
        else:
            num_windows = math.ceil((B - window_size) / max(1, window_size - overlap)) + 1 if B > window_size else 1
            total_steps = num_inference_steps * num_windows
            
        self.start_progress(total_steps)
        
        # Run the pipeline with tiling parameters
        with torch.inference_mode():
            result = pipe(
                images,
                height=height,
                width=width,
                output_type="pt",
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=False,
                progress_callback=self.update_progress,
                # AMD/Tiling parameters
                enable_spatial_tiling=enable_spatial_tiling,
                spatial_tile_size=spatial_tile_size,
                spatial_tile_overlap=spatial_tile_overlap,
                vae_encode_chunk_size=vae_encode_chunk_size,
                vae_decode_chunk_size=vae_decode_chunk_size,
                max_memory_usage_gb=max_memory_usage_gb,
            )
            
        res = result.frames[0]  # [B, H, W, C] 或 [B, C, H, W] 視 video_processor 輸出
        
        # 確保格式正確
        if res.dim() == 4 and res.shape[1] == 3:
            # [B, C, H, W] -> [B, H, W, C]
            res = res.permute(0, 2, 3, 1)
        
        # 檢查並修復無效值
        if torch.isnan(res).any() or torch.isinf(res).any():
            print("[DepthCrafter] 警告: 輸出包含無效值，正在修復...")
            res = torch.nan_to_num(res, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Convert to grayscale depth map
        if res.shape[-1] == 3:
            res = res.mean(dim=-1)  # [B, H, W]
        else:
            res = res.squeeze(-1)  # 移除最後一維如果是 1
        
        # Normalize depth maps
        res_min = res.min()
        res_max = res.max()
        if res_max - res_min > 1e-8:
            res = (res - res_min) / (res_max - res_min)
        else:
            print("[DepthCrafter] 警告: 深度圖幾乎沒有變化，可能有問題")
            res = torch.ones_like(res) * 0.5
        
        # Convert back to tensor with 3 channels
        depth_maps = res.unsqueeze(-1).repeat(1, 1, 1, 3)  # [B, H, W, 3]
        depth_maps = depth_maps.float().cpu()
        
        self.end_progress()
        
        return (depth_maps,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadDepthCrafterModel": DownloadAndLoadDepthCrafterModel,
    "DepthCrafter": DepthCrafter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadDepthCrafterModel": "Download And Load DepthCrafter Model",
    "DepthCrafter": "DepthCrafter",
}
