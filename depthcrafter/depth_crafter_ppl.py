from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    StableVideoDiffusionPipelineOutput,
    StableVideoDiffusionPipeline,
    retrieve_timesteps,
)
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)


class DepthCrafterPipeline(StableVideoDiffusionPipeline):
    """
    DepthCrafter Pipeline - AMD gfx1151 優化版本
    支援 2K+ 解析度原生處理 (Spatial Tiling)
    
    新增參數可透過 __call__ 傳入：
    - enable_spatial_tiling: 是否啟用空間分塊
    - spatial_tile_size: 空間分塊大小
    - spatial_tile_overlap: 空間分塊重疊
    - vae_tile_size: VAE 分塊大小
    - vae_tile_overlap: VAE 分塊重疊
    - vae_encode_chunk_size: VAE encoding 每批幀數
    - vae_decode_chunk_size: VAE decoding 每批幀數
    """
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)
    
    @classmethod
    def from_single_file(cls, *args, **kwargs):
        return super().from_single_file(*args, **kwargs)

    def _setup_for_amd(self):
        """設定 AMD 優化選項"""
        try:
            self.unet.set_default_attn_processor()
            self.vae.set_default_attn_processor()
            print("[AMD] 已啟用 Default Attention Processor")
        except Exception as e:
            print(f"[AMD] 警告: {e}")

    def _enable_vae_tiling(self, vae_tile_size: int = 256, vae_tile_overlap: int = 64):
        """啟用 VAE tiling"""
        try:
            self.vae.enable_tiling()
            if hasattr(self.vae, 'tile_sample_min_size'):
                self.vae.tile_sample_min_size = vae_tile_size
            if hasattr(self.vae, 'tile_latent_min_size'):
                self.vae.tile_latent_min_size = vae_tile_size // 8
            if hasattr(self.vae, 'tile_overlap_factor'):
                self.vae.tile_overlap_factor = vae_tile_overlap / vae_tile_size
        except Exception as e:
            print(f"[VAE Tiling] 警告: {e}")

    def _disable_vae_tiling(self):
        """停用 VAE tiling"""
        try:
            self.vae.disable_tiling()
        except:
            pass

    def _calculate_tiles(
        self, 
        height: int, 
        width: int, 
        tile_size: int, 
        tile_overlap: int
    ) -> List[Dict]:
        """
        計算空間分塊的座標
        
        Args:
            height: 影片高度
            width: 影片寬度
            tile_size: 分塊大小
            tile_overlap: 分塊重疊
        
        Returns:
            List of dicts with tile coordinates and padding info
        """
        stride = tile_size - tile_overlap
        
        tiles = []
        
        # 計算需要多少 tiles
        n_tiles_y = max(1, (height - tile_overlap + stride - 1) // stride)
        n_tiles_x = max(1, (width - tile_overlap + stride - 1) // stride)
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                y_start = ty * stride
                x_start = tx * stride
                y_end = min(y_start + tile_size, height)
                x_end = min(x_start + tile_size, width)
                
                # 確保 tile 尺寸是 64 的倍數
                actual_h = y_end - y_start
                actual_w = x_end - x_start
                
                # 計算需要的 padding 使其成為 64 的倍數
                target_h = ((actual_h + 63) // 64) * 64
                target_w = ((actual_w + 63) // 64) * 64
                
                pad_bottom = target_h - actual_h
                pad_right = target_w - actual_w
                
                tiles.append({
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start': x_start,
                    'x_end': x_end,
                    'pad_top': 0,
                    'pad_bottom': pad_bottom,
                    'pad_left': 0,
                    'pad_right': pad_right,
                    'target_h': target_h,
                    'target_w': target_w,
                    'tile_y': ty,
                    'tile_x': tx,
                })
        
        return tiles, n_tiles_y, n_tiles_x

    def _create_blend_mask(
        self, 
        tile_h: int, 
        tile_w: int, 
        is_top: bool, 
        is_bottom: bool, 
        is_left: bool, 
        is_right: bool,
        overlap: int
    ) -> torch.Tensor:
        """
        創建用於混合 tiles 的漸變遮罩
        在重疊區域使用線性漸變，確保無縫拼接
        """
        mask = torch.ones(tile_h, tile_w)
        
        # 上邊漸變
        if not is_top and overlap > 0:
            for i in range(min(overlap, tile_h)):
                mask[i, :] *= i / overlap
        
        # 下邊漸變
        if not is_bottom and overlap > 0:
            for i in range(min(overlap, tile_h)):
                mask[tile_h - 1 - i, :] *= i / overlap
        
        # 左邊漸變
        if not is_left and overlap > 0:
            for i in range(min(overlap, tile_w)):
                mask[:, i] *= i / overlap
        
        # 右邊漸變
        if not is_right and overlap > 0:
            for i in range(min(overlap, tile_w)):
                mask[:, tile_w - 1 - i] *= i / overlap
        
        return mask

    @torch.inference_mode()
    def encode_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        """編碼影片為 CLIP embeddings"""
        video_224 = _resize_with_antialiasing(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0

        embeddings = []
        for i in range(0, video_224.shape[0], chunk_size):
            tmp = self.feature_extractor(
                images=video_224[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            embeddings.append(self.image_encoder(tmp).image_embeds)
            torch.cuda.empty_cache()

        return torch.cat(embeddings, dim=0)

    @torch.inference_mode()
    def encode_vae_video_tiled(
        self,
        video: torch.Tensor,
        chunk_size: int = 1,
        vae_tile_size: int = 256,
        vae_tile_overlap: int = 64,
    ):
        """使用 tiling 編碼影片"""
        self._enable_vae_tiling(vae_tile_size, vae_tile_overlap)
        
        video_latents = []
        for i in range(0, video.shape[0], chunk_size):
            torch.cuda.empty_cache()
            chunk = video[i : i + chunk_size]
            latent = self.vae.encode(chunk).latent_dist.mode()
            video_latents.append(latent)
        
        self._disable_vae_tiling()
        return torch.cat(video_latents, dim=0)

    def _process_single_tile(
        self,
        video_tile: torch.Tensor,      # [T, C, H, W]
        num_inference_steps: int,
        guidance_scale: float,
        window_size: int,
        overlap: int,
        noise_aug_strength: float,
        generator: Optional[torch.Generator],
        device: torch.device,
        vae_tile_size: int,
        vae_tile_overlap: int,
        vae_encode_chunk_size: int,
        vae_decode_chunk_size: int,
        progress_callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        處理單一空間 tile
        這是核心處理函數，與原始 DepthCrafter 邏輯相同
        """
        num_frames = video_tile.shape[0]
        height, width = video_tile.shape[2], video_tile.shape[3]
        
        if num_frames <= window_size:
            window_size = num_frames
            overlap = 0
        stride = window_size - overlap
        
        # CLIP encoding
        video_embeddings = self.encode_video(
            video_tile, chunk_size=4
        ).unsqueeze(0)
        
        torch.cuda.empty_cache()
        
        # Add noise
        noise = randn_tensor(
            video_tile.shape, generator=generator, device=device, dtype=video_tile.dtype
        )
        video_noised = video_tile + noise_aug_strength * noise
        
        # VAE encoding
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        
        video_latents = self.encode_vae_video_tiled(
            video_noised.to(self.vae.dtype),
            chunk_size=vae_encode_chunk_size,
            vae_tile_size=vae_tile_size,
            vae_tile_overlap=vae_tile_overlap,
        ).unsqueeze(0)
        
        torch.cuda.empty_cache()
        
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        
        # Time IDs
        added_time_ids = self._get_add_time_ids(
            7, 127, noise_aug_strength,
            video_embeddings.dtype, 1, 1, False,
        ).to(device)
        
        # Timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        # Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents_init = self.prepare_latents(
            1, window_size, num_channels_latents,
            height, width, video_embeddings.dtype, device,
            generator, None,
        )
        latents_all = None
        
        idx_start = 0
        if overlap > 0:
            weights = torch.linspace(0, 1, overlap, device=device).view(1, overlap, 1, 1, 1)
        else:
            weights = None
        
        # Denoising loop
        while idx_start < num_frames - max(overlap, 0):
            idx_end = min(idx_start + window_size, num_frames)
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            
            latents = latents_init[:, :idx_end - idx_start].clone()
            if overlap > 0:
                latents_init = torch.cat(
                    [latents_init[:, -overlap:], latents_init[:, :stride]], dim=1
                )
            
            video_latents_current = video_latents[:, idx_start:idx_end]
            video_embeddings_current = video_embeddings[:, idx_start:idx_end]
            
            for i, t in enumerate(timesteps):
                if latents_all is not None and i == 0 and overlap > 0:
                    latents[:, :overlap] = (
                        latents_all[:, -overlap:]
                        + latents[:, :overlap] / self.scheduler.init_noise_sigma
                        * self.scheduler.sigmas[i]
                    )
                
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                latent_model_input = torch.cat(
                    [latent_model_input, video_latents_current], dim=2
                )
                
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=video_embeddings_current,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]
                
                if self.do_classifier_free_guidance:
                    latent_model_input = self.scheduler.scale_model_input(latents, t)
                    latent_model_input = torch.cat(
                        [latent_model_input, torch.zeros_like(latent_model_input)], dim=2
                    )
                    noise_pred_uncond = self.unet(
                        latent_model_input, t,
                        encoder_hidden_states=torch.zeros_like(video_embeddings_current),
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                # Progress callback
                if progress_callback is not None:
                    progress_callback(i)
            
            if latents_all is None:
                latents_all = latents.clone()
            else:
                if weights is not None and overlap > 0:
                    latents_all[:, -overlap:] = (
                        latents[:, :overlap] * weights 
                        + latents_all[:, -overlap:] * (1 - weights)
                    )
                latents_all = torch.cat([latents_all, latents[:, overlap:]], dim=1)
            
            idx_start += stride
            torch.cuda.empty_cache()
        
        # VAE decode
        self._enable_vae_tiling(vae_tile_size, vae_tile_overlap)
        latents_all = latents_all.to(dtype=self.vae.dtype)
        frames = self.decode_latents(latents_all, num_frames, vae_decode_chunk_size)
        self._disable_vae_tiling()
        
        return frames  # [1, T, C, H, W]

    @staticmethod
    def check_inputs(video, height, width):
        if not isinstance(video, torch.Tensor) and not isinstance(video, np.ndarray):
            raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(video)}")

    @torch.no_grad()
    def __call__(
        self,
        video: Union[np.ndarray, torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.0,
        window_size: Optional[int] = 110,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        overlap: int = 25,
        track_time: bool = False,
        progress_callback: Optional[Callable] = None,
        # ==================== AMD/Tiling 可配置參數 ====================
        enable_spatial_tiling: bool = True,
        spatial_tile_size: int = 768,
        spatial_tile_overlap: int = 128,
        vae_tile_size: int = 256,
        vae_tile_overlap: int = 64,
        vae_encode_chunk_size: int = 1,
        vae_decode_chunk_size: int = 2,
        # ==============================================================
    ):
        """
        DepthCrafter Pipeline - 支援 2K+ 原生解析度
        使用 Spatial Tiling 技術分塊處理
        
        Args:
            video: 輸入影片 [T, H, W, C] (np.ndarray) 或 [T, C, H, W] (torch.Tensor)
            height: 目標高度
            width: 目標寬度
            num_inference_steps: 去噪步數
            guidance_scale: CFG scale
            window_size: 時間滑動窗口大小
            overlap: 時間窗口重疊
            ...
            
            enable_spatial_tiling: 是否啟用空間分塊 (2K+ 必須)
            spatial_tile_size: 空間分塊大小 (像素，必須是 64 的倍數)
            spatial_tile_overlap: 空間分塊重疊 (像素)
            vae_tile_size: VAE 分塊大小
            vae_tile_overlap: VAE 分塊重疊
            vae_encode_chunk_size: VAE encoding 每批幀數
            vae_decode_chunk_size: VAE decoding 每批幀數
        """
        
        # AMD 優化設定
        self._setup_for_amd()
        self._guidance_scale = guidance_scale
        
        # 處理輸入
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # 確保是 64 的倍數
        height = ((height + 63) // 64) * 64
        width = ((width + 63) // 64) * 64
        
        # 確保 spatial_tile_size 是 64 的倍數
        spatial_tile_size = ((spatial_tile_size + 63) // 64) * 64
        
        self.check_inputs(video, height, width)
        
        device = self._execution_device
        
        # 轉換影片格式
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        video = video.to(device=device, dtype=self.dtype)
        
        # Resize 到目標尺寸
        if video.shape[2] != height or video.shape[3] != width:
            video = _resize_with_antialiasing(video, (height, width))
        
        video = video * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        num_frames = video.shape[0]
        
        print(f"[DepthCrafter] 輸入: {num_frames} frames @ {width}x{height}")
        print(f"[DepthCrafter] Tiling 配置:")
        print(f"  - enable_spatial_tiling: {enable_spatial_tiling}")
        print(f"  - spatial_tile_size: {spatial_tile_size}")
        print(f"  - spatial_tile_overlap: {spatial_tile_overlap}")
        print(f"  - vae_tile_size: {vae_tile_size}")
        print(f"  - vae_encode_chunk_size: {vae_encode_chunk_size}")
        print(f"  - vae_decode_chunk_size: {vae_decode_chunk_size}")
        
        # ==================== Spatial Tiling ====================
        # 判斷是否需要 tiling
        needs_tiling = enable_spatial_tiling and (height > spatial_tile_size or width > spatial_tile_size)
        
        if not needs_tiling:
            # 不需要 spatial tiling，直接處理
            print(f"[DepthCrafter] 解析度較小或已停用 tiling，直接處理")
            frames = self._process_single_tile(
                video, 
                num_inference_steps, 
                guidance_scale,
                window_size, 
                overlap, 
                noise_aug_strength,
                generator, 
                device,
                vae_tile_size,
                vae_tile_overlap,
                vae_encode_chunk_size,
                vae_decode_chunk_size,
                progress_callback,
            )
        else:
            # 需要 spatial tiling
            tiles, n_tiles_y, n_tiles_x = self._calculate_tiles(
                height, width, spatial_tile_size, spatial_tile_overlap
            )
            print(f"[DepthCrafter] 使用 Spatial Tiling: {len(tiles)} tiles ({n_tiles_x}x{n_tiles_y})")
            
            # 準備輸出 buffer 和 weight buffer
            output_buffer = torch.zeros(
                (1, num_frames, 3, height, width), 
                device=device, dtype=self.dtype
            )
            weight_buffer = torch.zeros(
                (1, num_frames, 1, height, width), 
                device=device, dtype=self.dtype
            )
            
            for tile_idx, tile_info in enumerate(tiles):
                ty = tile_info['tile_y']
                tx = tile_info['tile_x']
                
                print(f"[DepthCrafter] 處理 Tile {tile_idx + 1}/{len(tiles)} "
                      f"(row {ty + 1}/{n_tiles_y}, col {tx + 1}/{n_tiles_x})")
                
                y_start = tile_info['y_start']
                y_end = tile_info['y_end']
                x_start = tile_info['x_start']
                x_end = tile_info['x_end']
                
                # 提取 tile
                video_tile = video[:, :, y_start:y_end, x_start:x_end]
                
                # Padding 到目標尺寸 (64 的倍數)
                if tile_info['pad_bottom'] > 0 or tile_info['pad_right'] > 0:
                    video_tile = F.pad(
                        video_tile, 
                        (0, tile_info['pad_right'], 0, tile_info['pad_bottom']),
                        mode='reflect'
                    )
                
                torch.cuda.empty_cache()
                
                # 處理 tile
                tile_result = self._process_single_tile(
                    video_tile, 
                    num_inference_steps, 
                    guidance_scale,
                    window_size, 
                    overlap, 
                    noise_aug_strength,
                    generator, 
                    device,
                    vae_tile_size,
                    vae_tile_overlap,
                    vae_encode_chunk_size,
                    vae_decode_chunk_size,
                    progress_callback,
                )
                
                # 移除 padding
                actual_h = y_end - y_start
                actual_w = x_end - x_start
                tile_result = tile_result[:, :, :, :actual_h, :actual_w]
                
                # 創建混合遮罩
                is_top = (ty == 0)
                is_bottom = (ty == n_tiles_y - 1)
                is_left = (tx == 0)
                is_right = (tx == n_tiles_x - 1)
                
                blend_mask = self._create_blend_mask(
                    actual_h, actual_w,
                    is_top, is_bottom, is_left, is_right,
                    spatial_tile_overlap
                ).to(device=device, dtype=self.dtype)
                
                blend_mask = blend_mask.view(1, 1, 1, actual_h, actual_w)
                
                # 累加到 buffer
                output_buffer[:, :, :, y_start:y_end, x_start:x_end] += tile_result * blend_mask
                weight_buffer[:, :, :, y_start:y_end, x_start:x_end] += blend_mask
                
                torch.cuda.empty_cache()
            
            # 正規化
            frames = output_buffer / (weight_buffer + 1e-8)
        # ========================================================
        
        # 後處理
        if output_type != "latent":
            frames = self.video_processor.postprocess_video(
                video=frames, output_type=output_type
            )
        
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return frames
        
        return StableVideoDiffusionPipelineOutput(frames=frames)
