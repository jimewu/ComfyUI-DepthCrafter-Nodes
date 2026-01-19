import os
import torch
import math
import gc
import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

from .depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter


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
下載並載入 DepthCrafter 模型 (V9)

【AMD AI MAX+ 395 建議設定】
• enable_model_cpu_offload: ✅ 開啟
• enable_sequential_cpu_offload: ❌ 關閉

【修復】
• 使用正確的方式載入 pipeline
• 避免 from_pretrained 覆寫問題
"""

    def load_model(self, enable_model_cpu_offload, enable_sequential_cpu_offload):
        # 延遲導入
        from .depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
        
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

        # 載入 UNet
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # 關鍵修復：先載入 base pipeline 再創建 DepthCrafterPipeline
        from diffusers import StableVideoDiffusionPipeline
        
        base_pipe = StableVideoDiffusionPipeline.from_pretrained(
            pretrain_path,
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
        )
        
        # 直接傳入組件創建 DepthCrafterPipeline
        pipe = DepthCrafterPipeline(
            vae=base_pipe.vae,
            image_encoder=base_pipe.image_encoder,
            unet=unet,
            scheduler=base_pipe.scheduler,
            feature_extractor=base_pipe.feature_extractor,
        )
        
        del base_pipe
        aggressive_cleanup()
        
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
                "num_inference_steps": ("INT", {
                    "default": 10, 
                    "min": 1, 
                    "max": 50,
                    "tooltip": "去噪步數。更多步數 = 更好品質但更慢。2K 建議 8-12。"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.2, 
                    "min": 1.0, 
                    "max": 5.0, 
                    "step": 0.1,
                    "tooltip": "CFG 強度。1.0 = 無 CFG，建議 1.0-1.5。"
                }),
            },
            "optional": {
                # ==================== 自動優化 ====================
                "auto_optimize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "自動計算最佳 tile/window 配置。強烈建議保持開啟。"
                }),
                "max_memory_gb": ("FLOAT", {
                    "default": 100.0,
                    "min": 30.0,
                    "max": 128.0,
                    "step": 5.0,
                    "tooltip": "最大可用 GPU 記憶體 (GB)。128GB 系統建議設為 100-110。"
                }),
                
                # ==================== 手動覆蓋 ====================
                "tile_size_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 768,
                    "step": 64,
                    "tooltip": "手動設定 tile 大小。0 = 自動。2K 建議 384-512。"
                }),
                "window_size_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "tooltip": "手動設定時序窗口大小。0 = 自動。2K 建議 5-8。"
                }),
                
                # ==================== 進階設定 ====================
                "attention_slice_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "tooltip": "UNet 注意力分片大小。1 = 最省記憶體但最慢。AMD 建議 1-2。"
                }),
                "vae_encode_chunk": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 4,
                    "tooltip": "VAE 編碼每批處理的幀數。2K 建議 1。"
                }),
                "vae_decode_chunk": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 4,
                    "tooltip": "VAE 解碼每批處理的幀數。2K 建議 1。"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_maps",)
    FUNCTION = "process"
    DESCRIPTION = """
DepthCrafter V9 - AMD AI MAX+ 395 優化版本

【關鍵修復】
• 啟用 UNet Attention Slicing 解決 OOM
• 準確的記憶體估算（基於注意力矩陣大小）
• 強制 VAE float32 確保數值穩定

【2K 影片建議設定】
• auto_optimize: ✅ 開啟
• max_memory_gb: 100
• attention_slice_size: 1 (最省記憶體)
• num_inference_steps: 8-12

【記憶體估算參考】
• tile=512, window=5: ~60 GB
• tile=448, window=6: ~40 GB
• tile=384, window=5: ~25 GB

【如果仍然 OOM】
• 減少 tile_size_override (設為 384)
• 減少 window_size_override (設為 5)
• 確保 attention_slice_size = 1
"""
    
    def process(
        self, 
        depthcrafter_model, 
        images, 
        num_inference_steps, 
        guidance_scale, 
        auto_optimize=True,
        max_memory_gb=100.0,
        tile_size_override=0,
        window_size_override=0,
        attention_slice_size=1,
        vae_encode_chunk=1,
        vae_decode_chunk=1,
    ):
        aggressive_cleanup()
        
        device = depthcrafter_model['device']
        pipe = depthcrafter_model['pipe']
        
        B, H, W, C = images.shape

        # 強制 64 的倍數
        width = round(W / 64) * 64
        height = round(H / 64) * 64
        width = max(64, width)
        height = max(64, height)

        if width != W or height != H:
            print(f"DepthCrafter: Resizing {W}x{H} -> {width}x{height}")
            images_for_resize = images.permute(0, 3, 1, 2)
            images_resized = torch.nn.functional.interpolate(
                images_for_resize,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
            images = images_resized.permute(0, 2, 3, 1)
            H, W = height, width

        # 轉換輸入
        images = images.permute(0, 3, 1, 2)
        images = images.to(device=device, dtype=torch.float32)
        images = torch.clamp(images, 0, 1)
        
        # 計算進度條步數
        estimated_tiles = 1
        if auto_optimize and (height > 448 or width > 448):
            # 粗略估算
            tile_size = tile_size_override if tile_size_override > 0 else 448
            overlap = tile_size // 6
            stride = tile_size - overlap
            n_tiles_x = max(1, math.ceil((width - overlap) / stride))
            n_tiles_y = max(1, math.ceil((height - overlap) / stride))
            estimated_tiles = n_tiles_x * n_tiles_y
        
        window_size = window_size_override if window_size_override > 0 else 6
        num_windows = max(1, math.ceil(B / window_size))
        total_steps = num_inference_steps * num_windows * estimated_tiles
        
        self.start_progress(total_steps)
        
        aggressive_cleanup()
        
        # 準備參數
        spatial_tile_size = tile_size_override if tile_size_override > 0 else None
        window_size_param = window_size_override if window_size_override > 0 else None
        
        with torch.inference_mode():
            result = pipe(
                images,
                height=height,
                width=width,
                output_type="pt",
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                window_size=window_size_param,
                window_overlap=None,
                track_time=False,
                progress_callback=self.update_progress,
                # Tiling 配置
                enable_spatial_tiling=True,
                spatial_tile_size=spatial_tile_size,
                spatial_tile_overlap=None,
                vae_encode_chunk_size=vae_encode_chunk,
                vae_decode_chunk_size=vae_decode_chunk,
                # 優化配置
                max_memory_usage_gb=max_memory_gb,
                auto_optimize=auto_optimize,
                prefer_fewer_tiles=True,
                attention_slice_size=attention_slice_size,
            )
            
        res = result.frames[0]
        
        # 處理輸出形狀
        print(f"[DepthCrafter] 輸出形狀: {res.shape}")
        
        if res.dim() == 5:
            res = res.squeeze(0)
        
        if res.dim() == 4:
            if res.shape[1] == 3:
                res = res.permute(0, 2, 3, 1)
        
        # 修復無效值
        if torch.isnan(res).any() or torch.isinf(res).any():
            print(f"[DepthCrafter] 修復無效值")
            res = torch.nan_to_num(res, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 轉換為灰階
        if res.dim() == 4 and res.shape[-1] == 3:
            res = res.mean(dim=-1)
        elif res.dim() == 4 and res.shape[-1] == 1:
            res = res.squeeze(-1)
        
        # 正規化
        res_min = res.min()
        res_max = res.max()
        print(f"[DepthCrafter] 深度範圍: [{res_min.item():.4f}, {res_max.item():.4f}]")
        
        if res_max - res_min > 1e-6:
            res = (res - res_min) / (res_max - res_min)
        else:
            print("[DepthCrafter] 警告: 深度變化很小")
            res = torch.ones_like(res) * 0.5
        
        # 確保是 3D
        if res.dim() == 2:
            res = res.unsqueeze(0)
        
        # 擴展為 RGB
        depth_maps = res.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        depth_maps = depth_maps.float().cpu()
        
        print(f"[DepthCrafter] 最終輸出: {depth_maps.shape}")
        
        self.end_progress()
        aggressive_cleanup()
        
        return (depth_maps,)


# ==================== Node 註冊 ====================
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadDepthCrafterModel": DownloadAndLoadDepthCrafterModel,
    "DepthCrafter": DepthCrafter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadDepthCrafterModel": "Download And Load DepthCrafter Model",
    "DepthCrafter": "DepthCrafter",
}
