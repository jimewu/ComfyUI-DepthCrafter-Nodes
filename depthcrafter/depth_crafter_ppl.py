from typing import Callable, Dict, List, Optional, Union
import math
import gc
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


def aggressive_memory_cleanup():
    """激進的記憶體清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def safe_tensor(tensor: torch.Tensor, name: str = "tensor", fix_value: float = 0.0) -> torch.Tensor:
    """確保 tensor 沒有 NaN/Inf"""
    if tensor is None:
        return tensor
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        print(f"[數值警告] {name}: NaN={nan_count}, Inf={inf_count} / {total} ({100*(nan_count+inf_count)/total:.2f}%)")
        tensor = torch.nan_to_num(tensor, nan=fix_value, posinf=fix_value, neginf=fix_value)
    
    return tensor


class DepthCrafterPipeline(StableVideoDiffusionPipeline):
    """
    DepthCrafter Pipeline - AMD AI MAX+ 395 優化版本 V9
    關鍵修復：啟用 UNet 注意力分片解決 OOM
    """

    def _setup_for_amd(self, attention_slice_size: int = 1):
        """設定 AMD 優化選項 - 關鍵：啟用注意力分片"""
        # 1. 使用預設注意力處理器（避免 Flash Attention 問題）
        try:
            self.unet.set_default_attn_processor()
            self.vae.set_default_attn_processor()
            print("[AMD] 已啟用 Default Attention Processor")
        except Exception as e:
            print(f"[AMD] 警告: 無法設定 attention processor: {e}")
        
        # 2. 關鍵：啟用注意力分片 - 這會大幅減少記憶體使用
        # slice_size=1 表示一次只處理一個 head，最省記憶體但最慢
        # slice_size="auto" 會自動選擇
        try:
            self.enable_attention_slicing(slice_size=attention_slice_size)
            print(f"[AMD] 已啟用 Attention Slicing (slice_size={attention_slice_size})")
        except Exception as e:
            print(f"[AMD] 警告: 無法啟用 attention slicing: {e}")
        
        # 3. VAE slicing
        try:
            if hasattr(self.vae, 'enable_slicing'):
                self.vae.enable_slicing()
                print("[AMD] 已啟用 VAE Slicing")
        except Exception as e:
            pass
        
        print("[AMD] 強制 VAE 使用 float32 以確保數值穩定性")

    def _estimate_memory_for_config(self, tile_size: int, window_size: int) -> float:
        """
        準確的記憶體估算 - 基於 UNet 時序注意力
        
        關鍵公式：
        - 序列長度 = window_size × (tile_size/8)²
        - 注意力矩陣 = 序列長度² × 4 bytes × num_heads
        """
        latent_size = tile_size // 8  # VAE 下採樣因子
        seq_len = window_size * latent_size * latent_size
        
        # 注意力矩陣記憶體 (GB)
        num_heads = 16  # SVD UNet 典型值
        attention_memory_gb = (seq_len ** 2) * 4 * num_heads / (1024 ** 3)
        
        # 模型權重 (~4 GB for fp16)
        model_memory_gb = 4.0
        
        # 激活值和中間張量 (約為注意力的 2 倍)
        activation_memory_gb = attention_memory_gb * 2.0
        
        # 影片幀和 latents
        frame_memory_gb = window_size * 3 * tile_size * tile_size * 4 / (1024 ** 3)
        latent_memory_gb = window_size * 4 * latent_size * latent_size * 4 / (1024 ** 3)
        
        total = attention_memory_gb + model_memory_gb + activation_memory_gb + frame_memory_gb + latent_memory_gb
        
        # 安全邊際 1.5x（考慮碎片和其他開銷）
        # 但如果啟用了 attention slicing，可以用較小的邊際
        return total * 1.3

    def _find_optimal_config(
        self,
        height: int,
        width: int,
        num_frames: int,
        available_memory_gb: float = 100.0,
        prefer_fewer_tiles: bool = True,
    ) -> dict:
        """
        尋找最優配置 - 使用準確的記憶體估算
        
        對於 2K (2304x1408) 和 128GB：
        - 推薦：tile=448, window=6, tiles=12
        """
        # 注意：這些是經過計算的安全值
        # tile_size 必須能被 64 整除
        tile_sizes = [576, 512, 448, 384, 320, 256]
        window_sizes = [8, 7, 6, 5, 4, 3]
        
        best_config = None
        min_tiles = float('inf')
        max_usable_tile = 0
        
        for tile_size in tile_sizes:
            for window_size in window_sizes:
                estimated_mem = self._estimate_memory_for_config(tile_size, window_size)
                
                # 只使用 80% 的可用記憶體，留餘地
                if estimated_mem > available_memory_gb * 0.80:
                    continue
                
                overlap = max(64, tile_size // 6)
                stride = tile_size - overlap
                
                n_tiles_x = max(1, math.ceil((width - overlap) / stride))
                n_tiles_y = max(1, math.ceil((height - overlap) / stride))
                total_tiles = n_tiles_x * n_tiles_y
                
                if prefer_fewer_tiles:
                    # 優先選擇較少的 tiles（較大的 tile_size）
                    if tile_size > max_usable_tile or (tile_size == max_usable_tile and total_tiles < min_tiles):
                        max_usable_tile = tile_size
                        min_tiles = total_tiles
                        best_config = {
                            'tile_size': tile_size,
                            'window_size': window_size,
                            'overlap': overlap,
                            'window_overlap': max(1, window_size // 4),
                            'n_tiles_x': n_tiles_x,
                            'n_tiles_y': n_tiles_y,
                            'tiles': total_tiles,
                            'estimated_memory': estimated_mem,
                        }
                        # 找到最大可用 tile 就停止搜索 window
                        break
        
        if best_config is None:
            # 最保守的配置
            print("[警告] 無法找到合適配置，使用最小值")
            best_config = {
                'tile_size': 256,
                'window_size': 3,
                'overlap': 64,
                'window_overlap': 1,
                'n_tiles_x': math.ceil(width / 192),
                'n_tiles_y': math.ceil(height / 192),
                'tiles': -1,
                'estimated_memory': 15.0,
            }
        
        return best_config

    def _calculate_tiles(
        self, 
        height: int, 
        width: int, 
        tile_size: int, 
        tile_overlap: int
    ) -> tuple:
        """計算空間分塊的座標"""
        stride = tile_size - tile_overlap
        
        n_tiles_y = max(1, math.ceil((height - tile_overlap) / stride))
        n_tiles_x = max(1, math.ceil((width - tile_overlap) / stride))
        
        tiles = []
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                y_start = ty * stride
                x_start = tx * stride
                y_end = min(y_start + tile_size, height)
                x_end = min(x_start + tile_size, width)
                
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
                    'pad_bottom': pad_bottom,
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
        """創建用於混合 tiles 的漸變遮罩 - 使用 cosine 混合"""
        MIN_WEIGHT = 0.01
        
        def create_1d_weight(size: int, is_start: bool, is_end: bool, overlap: int) -> torch.Tensor:
            weight = torch.ones(size)
            
            if not is_start and overlap > 0:
                blend_len = min(overlap, size)
                t = torch.linspace(0, math.pi / 2, blend_len)
                blend = torch.sin(t) ** 2
                blend = blend * (1 - MIN_WEIGHT) + MIN_WEIGHT
                weight[:blend_len] = blend
                
            if not is_end and overlap > 0:
                blend_len = min(overlap, size)
                t = torch.linspace(0, math.pi / 2, blend_len)
                blend = torch.cos(t) ** 2
                blend = blend * (1 - MIN_WEIGHT) + MIN_WEIGHT
                weight[-blend_len:] = weight[-blend_len:] * blend
                
            return weight
        
        weight_y = create_1d_weight(tile_h, is_top, is_bottom, overlap)
        weight_x = create_1d_weight(tile_w, is_left, is_right, overlap)
        
        mask = weight_y.unsqueeze(1) * weight_x.unsqueeze(0)
        mask = torch.clamp(mask, min=MIN_WEIGHT)
        
        return mask

    @torch.inference_mode()
    def encode_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 2,
    ) -> torch.Tensor:
        """編碼影片為 CLIP embeddings"""
        video_224 = _resize_with_antialiasing(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0
        video_224 = torch.clamp(video_224, 0, 1)

        embeddings = []
        for i in range(0, video_224.shape[0], chunk_size):
            aggressive_memory_cleanup()
            
            chunk = video_224[i : i + chunk_size]
            
            tmp = self.feature_extractor(
                images=chunk,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=torch.float32)
            
            emb = self.image_encoder(tmp.half()).image_embeds
            emb = safe_tensor(emb.float(), f"CLIP embedding chunk {i}")
            embeddings.append(emb)
            del tmp, chunk

        result = torch.cat(embeddings, dim=0)
        del embeddings
        aggressive_memory_cleanup()
        return result

    @torch.inference_mode()
    def encode_vae_video_chunked(
        self,
        video: torch.Tensor,
        chunk_size: int = 1,
    ) -> torch.Tensor:
        """分塊編碼影片為 VAE latents - 強制使用 float32"""
        video_latents = []
        total_frames = video.shape[0]
        
        original_vae_dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        
        for i in range(0, total_frames, chunk_size):
            aggressive_memory_cleanup()
            
            chunk = video[i : i + chunk_size].float()
            chunk = torch.clamp(chunk, -1.0, 1.0)
            chunk = safe_tensor(chunk, f"VAE encode input chunk {i}")
            
            latent_dist = self.vae.encode(chunk).latent_dist
            latent = latent_dist.mode()
            
            latent = safe_tensor(latent, f"VAE latent chunk {i}")
            
            video_latents.append(latent.cpu())  # 移到 CPU 節省 GPU 記憶體
            del chunk, latent_dist, latent
        
        self.vae.to(dtype=original_vae_dtype)
        
        result = torch.cat(video_latents, dim=0)
        del video_latents
        aggressive_memory_cleanup()
        
        return result

    def _decode_latents_chunked(
        self,
        latents: torch.Tensor,
        num_frames: int,
        decode_chunk_size: int = 1,
    ) -> torch.Tensor:
        """分塊解碼 latents - 強制使用 float32"""
        device = latents.device
        batch_size = latents.shape[0]
        latent_height = latents.shape[3]
        latent_width = latents.shape[4]
        
        latents_scaled = latents.float() / self.vae.config.scaling_factor
        latents_scaled = safe_tensor(latents_scaled, "latents before decode")
        
        latents_flat = latents_scaled.reshape(-1, latents_scaled.shape[2], latent_height, latent_width)
        total_frames_to_decode = latents_flat.shape[0]
        
        original_vae_dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        
        decoded_frames = []
        
        for i in range(0, total_frames_to_decode, decode_chunk_size):
            aggressive_memory_cleanup()
            
            chunk = latents_flat[i : i + decode_chunk_size].float().to(device)
            chunk = safe_tensor(chunk, f"decode input chunk {i}")
            
            num_frames_in_chunk = chunk.shape[0]
            
            try:
                decoded = self.vae.decode(chunk, num_frames=num_frames_in_chunk).sample
            except TypeError:
                decoded = self.vae.decode(chunk).sample
            
            decoded = safe_tensor(decoded, f"decoded chunk {i}")
            decoded_frames.append(decoded.cpu())
            del decoded, chunk
        
        self.vae.to(dtype=original_vae_dtype)
        
        frames = torch.cat(decoded_frames, dim=0)
        del decoded_frames
        aggressive_memory_cleanup()
        
        output_height = frames.shape[2]
        output_width = frames.shape[3]
        frames = frames.reshape(batch_size, num_frames, 3, output_height, output_width)
        
        frames = (frames / 2 + 0.5)
        frames = torch.clamp(frames, 0, 1)
        frames = safe_tensor(frames, "final decoded frames", fix_value=0.5)
        
        return frames

    def _process_single_tile(
        self,
        video_tile: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        window_size: int,
        window_overlap: int,
        noise_aug_strength: float,
        generator: Optional[torch.Generator],
        device: torch.device,
        vae_encode_chunk_size: int,
        vae_decode_chunk_size: int,
        progress_callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """處理單一空間 tile"""
        num_frames = video_tile.shape[0]
        height, width = video_tile.shape[2], video_tile.shape[3]
        
        video_tile = video_tile.float()
        video_tile = torch.clamp(video_tile, -1.0, 1.0)
        video_tile = safe_tensor(video_tile, "input video_tile")
        
        if num_frames <= window_size:
            actual_window_size = num_frames
            actual_overlap = 0
        else:
            actual_window_size = window_size
            actual_overlap = window_overlap
        stride = max(1, actual_window_size - actual_overlap)
        
        # CLIP 編碼
        aggressive_memory_cleanup()
        video_embeddings = self.encode_video(video_tile, chunk_size=1).unsqueeze(0)
        video_embeddings = safe_tensor(video_embeddings, "video_embeddings")
        
        aggressive_memory_cleanup()
        
        # 添加噪聲
        noise = randn_tensor(
            video_tile.shape, generator=generator, device=device, dtype=torch.float32
        )
        video_noised = video_tile + noise_aug_strength * noise
        video_noised = torch.clamp(video_noised, -1.0, 1.0)
        del noise
        
        # VAE 編碼
        video_latents = self.encode_vae_video_chunked(
            video_noised,
            chunk_size=vae_encode_chunk_size,
        ).unsqueeze(0).to(device)
        video_latents = safe_tensor(video_latents, "video_latents")
        
        del video_noised
        aggressive_memory_cleanup()
        
        # Time IDs
        added_time_ids = self._get_add_time_ids(
            7, 127, noise_aug_strength,
            torch.float16, 1, 1, False,
        ).to(device)
        
        # Timesteps
        timesteps, num_inference_steps_actual = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        
        # Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents_init = self.prepare_latents(
            1, actual_window_size, num_channels_latents,
            height, width, torch.float16, device,
            generator, None,
        )
        
        latents_all = None
        
        idx_start = 0
        if actual_overlap > 0:
            weights = torch.linspace(0, 1, actual_overlap, device=device).view(1, actual_overlap, 1, 1, 1)
        else:
            weights = None
        
        window_count = 0
        while idx_start < num_frames:
            idx_end = min(idx_start + actual_window_size, num_frames)
            current_len = idx_end - idx_start
            
            if current_len == 0:
                break
            
            window_count += 1
            
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            
            latents = latents_init[:, :current_len].clone()
            
            if actual_overlap > 0 and stride > 0:
                latents_init = torch.cat(
                    [latents_init[:, -actual_overlap:], latents_init[:, :stride]], dim=1
                )
            
            video_latents_current = video_latents[:, idx_start:idx_end].half()
            video_embeddings_current = video_embeddings[:, idx_start:idx_end].half()
            
            # Denoising loop
            for i, t in enumerate(timesteps):
                # 每 3 步清理一次記憶體
                if i % 3 == 0:
                    aggressive_memory_cleanup()
                
                if latents_all is not None and i == 0 and actual_overlap > 0:
                    latents[:, :actual_overlap] = (
                        latents_all[:, -actual_overlap:].to(device)
                        + latents[:, :actual_overlap] / self.scheduler.init_noise_sigma
                        * self.scheduler.sigmas[i]
                    )
                
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                latent_model_input = torch.cat(
                    [latent_model_input, video_latents_current], dim=2
                )
                
                # UNet 前向傳播 - 注意力分片已在 _setup_for_amd 中啟用
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=video_embeddings_current,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]
                
                if torch.isnan(noise_pred).any():
                    print(f"[警告] Step {i}: UNet 輸出包含 NaN")
                    noise_pred = torch.nan_to_num(noise_pred, nan=0.0)
                
                del latent_model_input
                
                # CFG (if enabled)
                if self.do_classifier_free_guidance:
                    latent_uncond = self.scheduler.scale_model_input(latents, t)
                    latent_uncond = torch.cat(
                        [latent_uncond, torch.zeros_like(latent_uncond)], dim=2
                    )
                    noise_pred_uncond = self.unet(
                        latent_uncond, t,
                        encoder_hidden_states=torch.zeros_like(video_embeddings_current),
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    del latent_uncond
                    
                    if torch.isnan(noise_pred_uncond).any():
                        noise_pred_uncond = torch.nan_to_num(noise_pred_uncond, nan=0.0)
                    
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                    del noise_pred_uncond
                
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                del noise_pred
                
                if progress_callback is not None:
                    progress_callback(i)
            
            # 合併 latents
            if latents_all is None:
                latents_all = latents.cpu()
            else:
                latents_cpu = latents.cpu()
                if weights is not None and actual_overlap > 0:
                    blend_len = min(actual_overlap, latents_cpu.shape[1])
                    blend_weights = weights[:, :blend_len].cpu()
                    latents_all[:, -blend_len:] = (
                        latents_cpu[:, :blend_len] * blend_weights 
                        + latents_all[:, -blend_len:] * (1 - blend_weights)
                    )
                if current_len > actual_overlap:
                    latents_all = torch.cat([latents_all, latents_cpu[:, actual_overlap:]], dim=1)
                del latents_cpu
            
            del latents, video_latents_current, video_embeddings_current
            idx_start += stride
            aggressive_memory_cleanup()
        
        del video_latents, video_embeddings
        aggressive_memory_cleanup()
        
        # VAE 解碼
        latents_all = safe_tensor(latents_all.float().to(device), "latents_all before decode")
        
        frames = self._decode_latents_chunked(
            latents_all,
            num_frames, 
            vae_decode_chunk_size
        )
        
        del latents_all
        aggressive_memory_cleanup()
        
        frames = frames.to(device=device)
        frames = safe_tensor(frames, "final tile frames", fix_value=0.5)
        
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
        window_size: Optional[int] = None,
        window_overlap: Optional[int] = None,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        overlap: int = None,  # deprecated, use window_overlap
        track_time: bool = False,
        progress_callback: Optional[Callable] = None,
        # ==================== 配置參數 ====================
        enable_spatial_tiling: bool = True,
        spatial_tile_size: int = None,
        spatial_tile_overlap: int = None,
        vae_encode_chunk_size: int = 1,
        vae_decode_chunk_size: int = 1,
        max_memory_usage_gb: float = 100.0,
        auto_optimize: bool = True,
        prefer_fewer_tiles: bool = True,
        attention_slice_size: int = 1,  # 關鍵：注意力分片大小
        # =================================================
    ):
        """
        DepthCrafter Pipeline V9
        
        關鍵修復：
        1. 啟用 UNet 注意力分片
        2. 準確的記憶體估算
        3. 數值穩定性
        """
        
        # 設定 AMD 優化 - 包含注意力分片
        self._setup_for_amd(attention_slice_size=attention_slice_size)
        self._guidance_scale = guidance_scale
        
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        orig_height, orig_width = height, width
        height = ((height + 63) // 64) * 64
        width = ((width + 63) // 64) * 64
        
        self.check_inputs(video, height, width)
        
        device = self._execution_device
        
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        video = video.to(device=device, dtype=torch.float32)
        
        if video.shape[2] != height or video.shape[3] != width:
            print(f"[DepthCrafter] Resize: {video.shape[3]}x{video.shape[2]} -> {width}x{height}")
            video = _resize_with_antialiasing(video, (height, width))
        
        video = torch.clamp(video, 0, 1)
        video = video * 2.0 - 1.0
        video = safe_tensor(video, "input video")
        
        num_frames = video.shape[0]
        
        # 自動優化配置
        if auto_optimize:
            optimal = self._find_optimal_config(
                height, width, num_frames, 
                max_memory_usage_gb,
                prefer_fewer_tiles
            )
            
            if spatial_tile_size is None:
                spatial_tile_size = optimal['tile_size']
            if spatial_tile_overlap is None:
                spatial_tile_overlap = optimal['overlap']
            if window_size is None:
                window_size = optimal['window_size']
            if window_overlap is None:
                window_overlap = optimal['window_overlap']
            
            print(f"[DepthCrafter] 自動優化配置:")
            print(f"  - tile_size: {spatial_tile_size}, tile_overlap: {spatial_tile_overlap}")
            print(f"  - window_size: {window_size}, window_overlap: {window_overlap}")
            print(f"  - tiles: {optimal['n_tiles_x']}x{optimal['n_tiles_y']} = {optimal['tiles']}")
            print(f"  - 預估記憶體: {optimal['estimated_memory']:.1f} GB")
            print(f"  - attention_slice_size: {attention_slice_size}")
        else:
            spatial_tile_size = spatial_tile_size or 448
            spatial_tile_overlap = spatial_tile_overlap or 96
            window_size = window_size or 6
            window_overlap = window_overlap or 2
        
        # backward compatibility
        if overlap is not None:
            window_overlap = overlap
        
        spatial_tile_size = ((spatial_tile_size + 63) // 64) * 64
        
        print(f"[DepthCrafter] 輸入: {num_frames} frames @ {width}x{height}")
        print(f"[DepthCrafter] VAE 使用 float32 | Attention Slicing 已啟用")
        
        needs_tiling = enable_spatial_tiling and (height > spatial_tile_size or width > spatial_tile_size)
        
        if not needs_tiling:
            print(f"[DepthCrafter] 直接處理 (無需 spatial tiling)")
            aggressive_memory_cleanup()
            frames = self._process_single_tile(
                video, 
                num_inference_steps, 
                guidance_scale,
                window_size, 
                window_overlap, 
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
            
            # 使用 CPU 作為中間緩衝區
            output_buffer = torch.zeros(
                (1, num_frames, 3, height, width), 
                device='cpu',
                dtype=torch.float32
            )
            weight_buffer = torch.zeros(
                (1, num_frames, 1, height, width), 
                device='cpu',
                dtype=torch.float32
            )
            
            for tile_idx, tile_info in enumerate(tiles):
                ty = tile_info['tile_y']
                tx = tile_info['tile_x']
                
                print(f"[DepthCrafter] Tile {tile_idx + 1}/{len(tiles)} "
                      f"(row {ty + 1}/{n_tiles_y}, col {tx + 1}/{n_tiles_x})")
                
                y_start = tile_info['y_start']
                y_end = tile_info['y_end']
                x_start = tile_info['x_start']
                x_end = tile_info['x_end']
                
                # 提取 tile
                video_tile = video[:, :, y_start:y_end, x_start:x_end].clone()
                
                # Padding
                if tile_info['pad_bottom'] > 0 or tile_info['pad_right'] > 0:
                    video_tile = F.pad(
                        video_tile, 
                        (0, tile_info['pad_right'], 0, tile_info['pad_bottom']),
                        mode='reflect'
                    )
                
                aggressive_memory_cleanup()
                
                # 處理 tile
                tile_result = self._process_single_tile(
                    video_tile, 
                    num_inference_steps, 
                    guidance_scale,
                    window_size, 
                    window_overlap, 
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
                
                tile_result_cpu = tile_result.float().cpu()
                
                del tile_result, video_tile
                aggressive_memory_cleanup()
                
                # 創建混合遮罩
                is_top = (ty == 0)
                is_bottom = (ty == n_tiles_y - 1)
                is_left = (tx == 0)
                is_right = (tx == n_tiles_x - 1)
                
                blend_mask = self._create_blend_mask(
                    actual_h, actual_w,
                    is_top, is_bottom, is_left, is_right,
                    spatial_tile_overlap
                )
                blend_mask = blend_mask.view(1, 1, 1, actual_h, actual_w)
                
                # 累加
                output_buffer[:, :, :, y_start:y_end, x_start:x_end] += tile_result_cpu * blend_mask
                weight_buffer[:, :, :, y_start:y_end, x_start:x_end] += blend_mask
                
                del tile_result_cpu, blend_mask
                aggressive_memory_cleanup()
            
            # 正規化
            weight_buffer = torch.clamp(weight_buffer, min=0.01)
            frames = output_buffer / weight_buffer
            
            frames = safe_tensor(frames, "blended frames", fix_value=0.5)
            frames = frames.to(device=device)
            
            del output_buffer, weight_buffer
            aggressive_memory_cleanup()
            
            frames = torch.clamp(frames, 0, 1)
        
        # 後處理
        if output_type != "latent":
            frames = self.video_processor.postprocess_video(
                video=frames, output_type=output_type
            )
        
        self.maybe_free_model_hooks()
        aggressive_memory_cleanup()
        
        if not return_dict:
            return frames
        
        return StableVideoDiffusionPipelineOutput(frames=frames)
