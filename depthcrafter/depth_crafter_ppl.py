from typing import Callable, Dict, List, Optional, Union
import math
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
    DepthCrafter Pipeline - AMD gfx1151 優化版本 V3
    支援 2K+ 解析度原生處理 (Spatial Tiling)
    修復版本：解決 OOM 和輸出全黑問題
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
            # 強制使用 Default Attention Processor，避免 HIPBLAS 錯誤
            self.unet.set_default_attn_processor()
            self.vae.set_default_attn_processor()
            print("[AMD] 已啟用 Default Attention Processor")
        except Exception as e:
            print(f"[AMD] 警告: 無法設定 attention processor: {e}")
        
        # 啟用 attention slicing 以減少記憶體使用
        try:
            if hasattr(self.unet, 'enable_attention_slicing'):
                self.unet.enable_attention_slicing(slice_size="auto")
            if hasattr(self.vae, 'enable_slicing'):
                self.vae.enable_slicing()
        except Exception as e:
            print(f"[AMD] 警告: 無法啟用 attention slicing: {e}")

    def _calculate_tiles(
        self, 
        height: int, 
        width: int, 
        tile_size: int, 
        tile_overlap: int
    ) -> tuple:
        """計算空間分塊的座標"""
        stride = tile_size - tile_overlap
        
        # 計算需要多少 tiles
        n_tiles_y = max(1, math.ceil((height - tile_overlap) / stride))
        n_tiles_x = max(1, math.ceil((width - tile_overlap) / stride))
        
        tiles = []
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                y_start = ty * stride
                x_start = tx * stride
                y_end = min(y_start + tile_size, height)
                x_end = min(x_start + tile_size, width)
                
                # 確保 tile 尺寸是 64 的倍數
                actual_h = y_end - y_start
                actual_w = x_end - x_start
                
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
        創建用於混合 tiles 的漸變遮罩，使用 cosine blending 確保數值穩定性
        修復：確保最小權重不為 0，避免 NaN
        """
        MIN_WEIGHT = 0.01  # 最小權重，避免除法產生 NaN
        
        def create_1d_weight(size: int, is_start: bool, is_end: bool, overlap: int) -> torch.Tensor:
            weight = torch.ones(size)
            
            if not is_start and overlap > 0:
                blend_len = min(overlap, size)
                # Cosine blend: 從 MIN_WEIGHT 到 1
                t = torch.linspace(0, math.pi / 2, blend_len)
                blend = torch.sin(t) ** 2
                blend = blend * (1 - MIN_WEIGHT) + MIN_WEIGHT
                weight[:blend_len] = blend
                
            if not is_end and overlap > 0:
                blend_len = min(overlap, size)
                # Cosine blend: 從 1 到 MIN_WEIGHT
                t = torch.linspace(0, math.pi / 2, blend_len)
                blend = torch.cos(t) ** 2
                blend = blend * (1 - MIN_WEIGHT) + MIN_WEIGHT
                weight[-blend_len:] = blend
                
            return weight
        
        weight_y = create_1d_weight(tile_h, is_top, is_bottom, overlap)
        weight_x = create_1d_weight(tile_w, is_left, is_right, overlap)
        
        # 外積創建 2D mask
        mask = weight_y.unsqueeze(1) * weight_x.unsqueeze(0)
        
        # 確保最小值
        mask = torch.clamp(mask, min=MIN_WEIGHT)
        
        return mask

    @torch.inference_mode()
    def encode_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 4,
    ) -> torch.Tensor:
        """編碼影片為 CLIP embeddings"""
        video_224 = _resize_with_antialiasing(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0

        embeddings = []
        for i in range(0, video_224.shape[0], chunk_size):
            torch.cuda.empty_cache()
            tmp = self.feature_extractor(
                images=video_224[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            embeddings.append(self.image_encoder(tmp).image_embeds)

        return torch.cat(embeddings, dim=0)

    @torch.inference_mode()
    def encode_vae_video_chunked(
        self,
        video: torch.Tensor,
        chunk_size: int = 1,
    ) -> torch.Tensor:
        """
        分塊編碼影片為 VAE latents
        輸入: [T, C, H, W] 範圍 [-1, 1]
        輸出: [T, latent_C, H/8, W/8]
        """
        video_latents = []
        total_frames = video.shape[0]
        
        for i in range(0, total_frames, chunk_size):
            torch.cuda.empty_cache()
            chunk = video[i : i + chunk_size]
            latent = self.vae.encode(chunk).latent_dist.mode()
            video_latents.append(latent)
            
        return torch.cat(video_latents, dim=0)

    def _decode_latents_chunked(
        self,
        latents: torch.Tensor,
        num_frames: int,
        decode_chunk_size: int = 2,
    ) -> torch.Tensor:
        """
        分塊解碼 latents 為影片幀
        
        Args:
            latents: [B, F, C, H/8, W/8] 格式的 latents
            num_frames: 幀數
            decode_chunk_size: 每批解碼的幀數
        
        Returns:
            frames: [B, F, C, H, W] 格式，範圍 [0, 1]
        """
        # latents shape: [B, F, C, H/8, W/8]
        batch_size = latents.shape[0]
        latent_height = latents.shape[3]
        latent_width = latents.shape[4]
        
        # 反標準化
        latents = latents / self.vae.config.scaling_factor
        
        # 轉換為 [B*F, C, H/8, W/8] 以便分塊處理
        latents_flat = latents.reshape(-1, latents.shape[2], latent_height, latent_width)
        total_frames_to_decode = latents_flat.shape[0]
        
        # 分塊解碼
        decoded_frames = []
        for i in range(0, total_frames_to_decode, decode_chunk_size):
            torch.cuda.empty_cache()
            chunk = latents_flat[i : i + decode_chunk_size]
            num_frames_in_chunk = chunk.shape[0]
            
            # AutoencoderKLTemporalDecoder 需要 num_frames 參數
            try:
                # 對於 temporal decoder，使用 num_frames 參數
                decoded = self.vae.decode(chunk, num_frames=num_frames_in_chunk).sample
            except TypeError:
                # 如果不支援 num_frames 參數（標準 VAE）
                decoded = self.vae.decode(chunk).sample
            
            decoded_frames.append(decoded)
        
        # 合併: [total_frames, C, H, W]
        frames = torch.cat(decoded_frames, dim=0)
        
        # 調整形狀回 [B, F, C, H, W]
        output_height = frames.shape[2]
        output_width = frames.shape[3]
        frames = frames.reshape(batch_size, num_frames, 3, output_height, output_width)
        
        # 正規化到 [0, 1]
        frames = (frames / 2 + 0.5).clamp(0, 1)
        
        return frames

    def _process_single_tile(
        self,
        video_tile: torch.Tensor,      # [T, C, H, W] 範圍 [-1, 1]
        num_inference_steps: int,
        guidance_scale: float,
        window_size: int,
        overlap: int,
        noise_aug_strength: float,
        generator: Optional[torch.Generator],
        device: torch.device,
        vae_encode_chunk_size: int,
        vae_decode_chunk_size: int,
        progress_callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        處理單一空間 tile
        
        Returns:
            frames: [1, T, C, H, W] 格式的深度圖，範圍 [0, 1]
        """
        num_frames = video_tile.shape[0]
        height, width = video_tile.shape[2], video_tile.shape[3]
        
        if num_frames <= window_size:
            actual_window_size = num_frames
            actual_overlap = 0
        else:
            actual_window_size = window_size
            actual_overlap = overlap
        stride = max(1, actual_window_size - actual_overlap)
        
        # CLIP encoding
        video_embeddings = self.encode_video(
            video_tile, chunk_size=4
        ).unsqueeze(0)  # [1, T, 1024]
        
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
        
        # [T, C, H/8, W/8]
        video_latents = self.encode_vae_video_chunked(
            video_noised.to(self.vae.dtype),
            chunk_size=vae_encode_chunk_size,
        ).unsqueeze(0)  # [1, T, C, H/8, W/8]
        
        torch.cuda.empty_cache()
        
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        
        # Time IDs
        added_time_ids = self._get_add_time_ids(
            7, 127, noise_aug_strength,
            video_embeddings.dtype, 1, 1, False,
        ).to(device)
        
        # Timesteps
        timesteps, num_inference_steps_actual = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        num_warmup_steps = len(timesteps) - num_inference_steps_actual * self.scheduler.order
        
        # Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents_init = self.prepare_latents(
            1, actual_window_size, num_channels_latents,
            height, width, video_embeddings.dtype, device,
            generator, None,
        )  # [1, window_size, C, H/8, W/8]
        
        latents_all = None
        
        idx_start = 0
        if actual_overlap > 0:
            weights = torch.linspace(0, 1, actual_overlap, device=device).view(1, actual_overlap, 1, 1, 1)
        else:
            weights = None
        
        # Denoising loop with temporal sliding window
        while idx_start < num_frames:
            idx_end = min(idx_start + actual_window_size, num_frames)
            current_len = idx_end - idx_start
            
            if current_len == 0:
                break
            
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            
            latents = latents_init[:, :current_len].clone()
            
            if actual_overlap > 0 and stride > 0:
                latents_init = torch.cat(
                    [latents_init[:, -actual_overlap:], latents_init[:, :stride]], dim=1
                )
            
            video_latents_current = video_latents[:, idx_start:idx_end]
            video_embeddings_current = video_embeddings[:, idx_start:idx_end]
            
            for i, t in enumerate(timesteps):
                if latents_all is not None and i == 0 and actual_overlap > 0:
                    latents[:, :actual_overlap] = (
                        latents_all[:, -actual_overlap:]
                        + latents[:, :actual_overlap] / self.scheduler.init_noise_sigma
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
                    latent_model_input_uncond = self.scheduler.scale_model_input(latents, t)
                    latent_model_input_uncond = torch.cat(
                        [latent_model_input_uncond, torch.zeros_like(latent_model_input_uncond)], dim=2
                    )
                    noise_pred_uncond = self.unet(
                        latent_model_input_uncond, t,
                        encoder_hidden_states=torch.zeros_like(video_embeddings_current),
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                if progress_callback is not None:
                    progress_callback(i)
            
            # 合併 latents
            if latents_all is None:
                latents_all = latents.clone()
            else:
                if weights is not None and actual_overlap > 0:
                    blend_len = min(actual_overlap, latents.shape[1])
                    blend_weights = weights[:, :blend_len]
                    latents_all[:, -blend_len:] = (
                        latents[:, :blend_len] * blend_weights 
                        + latents_all[:, -blend_len:] * (1 - blend_weights)
                    )
                if current_len > actual_overlap:
                    latents_all = torch.cat([latents_all, latents[:, actual_overlap:]], dim=1)
            
            idx_start += stride
            torch.cuda.empty_cache()
        
        # VAE decode
        latents_all = latents_all.to(dtype=self.vae.dtype)
        
        # 使用分塊解碼
        frames = self._decode_latents_chunked(
            latents_all, 
            num_frames, 
            vae_decode_chunk_size
        )  # [1, T, C, H, W]
        
        # 檢查輸出有效性
        if torch.isnan(frames).any() or torch.isinf(frames).any():
            print("[警告] 解碼結果包含 NaN 或 Inf，嘗試修復...")
            frames = torch.nan_to_num(frames, nan=0.5, posinf=1.0, neginf=0.0)
        
        return frames

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
        spatial_tile_size: int = 1024,      # 增大預設值以更好利用記憶體
        spatial_tile_overlap: int = 192,    # 增大重疊減少接縫
        vae_encode_chunk_size: int = 2,     # 增大以提高效率
        vae_decode_chunk_size: int = 4,     # 增大以提高效率
        max_memory_usage_gb: float = 100.0, # 最大記憶體使用量 (GB)
        # ==============================================================
    ):
        """
        DepthCrafter Pipeline - 支援 2K+ 原生解析度
        
        AMD gfx1151 優化版本，針對 128GB 統一記憶體設計
        """
        
        # AMD 優化設定
        self._setup_for_amd()
        self._guidance_scale = guidance_scale
        
        # 處理輸入
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # 確保是 64 的倍數
        orig_height, orig_width = height, width
        height = ((height + 63) // 64) * 64
        width = ((width + 63) // 64) * 64
        
        # 確保 spatial_tile_size 是 64 的倍數
        spatial_tile_size = ((spatial_tile_size + 63) // 64) * 64
        
        self.check_inputs(video, height, width)
        
        device = self._execution_device
        
        # 轉換影片格式
        if isinstance(video, np.ndarray):
            # np.ndarray 假設為 [T, H, W, C]
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        video = video.to(device=device, dtype=self.dtype)
        
        # Resize 到目標尺寸 (如果需要)
        if video.shape[2] != height or video.shape[3] != width:
            if orig_height != height or orig_width != width:
                print(f"[DepthCrafter] Resize: {video.shape[3]}x{video.shape[2]} -> {width}x{height}")
            video = _resize_with_antialiasing(video.float(), (height, width)).to(self.dtype)
        
        video = video * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        num_frames = video.shape[0]
        
        print(f"[DepthCrafter] 輸入: {num_frames} frames @ {width}x{height}")
        print(f"[DepthCrafter] 配置:")
        print(f"  - enable_spatial_tiling: {enable_spatial_tiling}")
        print(f"  - spatial_tile_size: {spatial_tile_size}")
        print(f"  - spatial_tile_overlap: {spatial_tile_overlap}")
        print(f"  - vae_encode_chunk_size: {vae_encode_chunk_size}")
        print(f"  - vae_decode_chunk_size: {vae_decode_chunk_size}")
        print(f"  - max_memory_usage_gb: {max_memory_usage_gb}")
        
        # ==================== Spatial Tiling ====================
        needs_tiling = enable_spatial_tiling and (height > spatial_tile_size or width > spatial_tile_size)
        
        if not needs_tiling:
            print(f"[DepthCrafter] 直接處理 (無需 spatial tiling)")
            frames = self._process_single_tile(
                video, 
                num_inference_steps, 
                guidance_scale,
                window_size, 
                overlap, 
                noise_aug_strength,
                generator, 
                device,
                vae_encode_chunk_size,
                vae_decode_chunk_size,
                progress_callback,
            )
        else:
            tiles, n_tiles_y, n_tiles_x = self._calculate_tiles(
                height, width, spatial_tile_size, spatial_tile_overlap
            )
            print(f"[DepthCrafter] Spatial Tiling: {len(tiles)} tiles ({n_tiles_x}x{n_tiles_y})")
            
            # 準備輸出 buffer: [1, F, C, H, W]
            output_buffer = torch.zeros(
                (1, num_frames, 3, height, width), 
                device=device, dtype=torch.float32  # 使用 float32 避免精度問題
            )
            weight_buffer = torch.zeros(
                (1, num_frames, 1, height, width), 
                device=device, dtype=torch.float32
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
                
                # 提取 tile: [T, C, H, W]
                video_tile = video[:, :, y_start:y_end, x_start:x_end].clone()
                
                # Padding 到 64 的倍數
                if tile_info['pad_bottom'] > 0 or tile_info['pad_right'] > 0:
                    video_tile = F.pad(
                        video_tile, 
                        (0, tile_info['pad_right'], 0, tile_info['pad_bottom']),
                        mode='reflect'
                    )
                
                torch.cuda.empty_cache()
                
                # 處理 tile -> [1, T, C, tile_H, tile_W]
                tile_result = self._process_single_tile(
                    video_tile, 
                    num_inference_steps, 
                    guidance_scale,
                    window_size, 
                    overlap, 
                    noise_aug_strength,
                    generator, 
                    device,
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
                ).to(device=device, dtype=torch.float32)
                
                # 擴展 blend_mask 為 [1, 1, 1, H, W]
                blend_mask = blend_mask.view(1, 1, 1, actual_h, actual_w)
                
                # 累加到 buffer (使用 float32 避免精度問題)
                tile_result_f32 = tile_result.float()
                output_buffer[:, :, :, y_start:y_end, x_start:x_end] += tile_result_f32 * blend_mask
                weight_buffer[:, :, :, y_start:y_end, x_start:x_end] += blend_mask
                
                # 清理 tile 記憶體
                del video_tile, tile_result, tile_result_f32, blend_mask
                torch.cuda.empty_cache()
            
            # 正規化 (避免除以 0)
            weight_buffer = torch.clamp(weight_buffer, min=0.01)
            frames = output_buffer / weight_buffer
            
            # 轉回 self.dtype
            frames = frames.to(self.dtype)
            
            # 檢查結果有效性
            if torch.isnan(frames).any() or torch.isinf(frames).any():
                print("[警告] 混合結果包含 NaN 或 Inf，嘗試修復...")
                frames = torch.nan_to_num(frames, nan=0.5, posinf=1.0, neginf=0.0)
            
            # 確保範圍在 [0, 1]
            frames = torch.clamp(frames, 0, 1)
            
            # 清理 buffer
            del output_buffer, weight_buffer
            torch.cuda.empty_cache()
        
        # ========================================================
        
        # 後處理
        if output_type != "latent":
            # frames 目前是 [1, F, C, H, W]
            frames = self.video_processor.postprocess_video(
                video=frames, output_type=output_type
            )
        
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return frames
        
        return StableVideoDiffusionPipelineOutput(frames=frames)
