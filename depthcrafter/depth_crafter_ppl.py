from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import cv2

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
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)
    
    @classmethod
    def from_single_file(cls, *args, **kwargs):
        return super().from_single_file(*args, **kwargs)

    @torch.inference_mode()
    def encode_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 2,  # 極限保守: 設為 2
    ) -> torch.Tensor:
        
        torch.cuda.empty_cache()

        # 標準化縮放
        video_224 = _resize_with_antialiasing(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        embeddings = []
        for i in range(0, video_224.shape[0], chunk_size):
            # 加上 dtype 轉換保護，防止混合精度錯誤
            sub_video = video_224[i : i + chunk_size].to(device=video.device, dtype=self.image_encoder.dtype)
            
            tmp = self.feature_extractor(
                images=sub_video,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=self.image_encoder.dtype)
            
            embeddings.append(self.image_encoder(tmp).image_embeds)
            torch.cuda.empty_cache()

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    @torch.inference_mode()
    def encode_vae_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 14, # 這裡保持稍大，但在外部控制
    ):
        torch.cuda.empty_cache()
        video_latents = []
        
        # ================= AMD Fix: VAE Tiling On Encode =================
        try:
            self.enable_vae_tiling()
            # 強制極小 Tile 避免 hipBLAS 錯誤
            self.vae.tile_sample_min_size = 256
            self.vae.tile_latent_min_size = 32
            self.vae.tile_overlap = 32 
        except:
            pass
        # ================================================================

        for i in range(0, video.shape[0], chunk_size):
            batch = video[i : i + chunk_size]
            video_latents.append(
                self.vae.encode(batch).latent_dist.mode()
            )
            torch.cuda.empty_cache() # 重要
            
        video_latents = torch.cat(video_latents, dim=0)
        return video_latents

    @staticmethod
    def check_inputs(video, height, width):
        if not isinstance(video, torch.Tensor) and not isinstance(video, np.ndarray):
            raise ValueError(
                f"Expected `video` to be a `torch.Tensor` or `VideoReader`, but got a {type(video)}"
            )
        pass 

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
        decode_chunk_size: Optional[int] = 2,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        overlap: int = 25,
        track_time: bool = False,
        progress_callback: Optional[Callable] = None,
    ):
        # 0. 強制對齊長寬為 64 的倍數 (這是 hipBLAS 運算的硬性要求)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        def make_multiple_of_64(val):
            return ((val + 63) // 64) * 64

        target_h = make_multiple_of_64(height)
        target_w = make_multiple_of_64(width)

        # 自動 Resize (保持原圖比例下的最小變動)
        if target_h != height or target_w != width:
             print(f"AMD Fix: Adjusting resolution slightly to {target_w}x{target_h} (multiple of 64 required)")
             if isinstance(video, torch.Tensor):
                 video = _resize_with_antialiasing(video, (target_h, target_w))
             elif isinstance(video, np.ndarray):
                 new_video = []
                 for frame in video:
                    new_video.append(cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4))
                 video = np.array(new_video)
             height, width = target_h, target_w

        num_frames = video.shape[0]
        decode_chunk_size = 2 # 強制極低 VAE 解碼 Chunk
        
        if num_frames <= window_size:
            window_size = num_frames
            overlap = 0
        stride = window_size - overlap

        self.check_inputs(video, height, width)

        # ================= AMD Fatal Error Fix: SLICED ATTENTION =================
        # 這是修復 hipblasSgemmStridedBatched 錯誤的唯一方法
        # 它將巨大的 Attention 矩陣切開，避免觸發底層數學庫的 Bug
        try:
             # 強制開啟切片注意力機制
             self.enable_sliced_attention()
             print("AMD Fix: Enabled Sliced Attention (Crucial for 2K resolution)")
             
             # 額外開啟 VAE Slicing
             self.enable_vae_slicing()
        except Exception as e:
             print(f"Warning: Could not enable sliced attention: {e}")
             
        # 強制清理記憶體，避免碎片化
        torch.cuda.empty_cache()
        # =========================================================================

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        self._guidance_scale = guidance_scale

        # 3. Encode input video
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        else:
            assert isinstance(video, torch.Tensor)
        video = video.to(device=device, dtype=self.dtype)
        video = video * 2.0 - 1.0 

        # 安全檢查 Check for NaNs
        if torch.isnan(video).any():
            print("Warning: Input video contains NaNs, fixing...")
            video = torch.nan_to_num(video, 0.0)

        video_embeddings = self.encode_video(
            video, chunk_size=2 
        ).unsqueeze(0)
        
        torch.cuda.empty_cache()

        # 4. Encode input image using VAE
        noise = randn_tensor(
            video.shape, generator=generator, device=device, dtype=video.dtype
        )
        video = video + noise_aug_strength * noise 

        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        # 編碼：使用極小 Chunk 大小
        video_latents = self.encode_vae_video(
            video.to(self.vae.dtype),
            chunk_size=2, 
        ).unsqueeze(0)

        torch.cuda.empty_cache()

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            7,
            127,
            noise_aug_strength,
            video_embeddings.dtype,
            batch_size,
            1,
            False,
        ) 
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents_init = self.prepare_latents(
            batch_size,
            window_size,
            num_channels_latents,
            height,
            width,
            video_embeddings.dtype,
            device,
            generator,
            latents,
        ) 
        latents_all = None

        idx_start = 0
        if overlap > 0:
            weights = torch.linspace(0, 1, overlap, device=device)
            weights = weights.view(1, overlap, 1, 1, 1)
        else:
            weights = None

        torch.cuda.empty_cache()

        while idx_start < num_frames - overlap:
            idx_end = min(idx_start + window_size, num_frames)
            self.scheduler.set_timesteps(num_inference_steps, device=device)

            # 9. Denoising loop
            latents = latents_init[:, : idx_end - idx_start].clone()
            latents_init = torch.cat(
                [latents_init[:, -overlap:], latents_init[:, :stride]], dim=1
            )

            video_latents_current = video_latents[:, idx_start:idx_end]
            video_embeddings_current = video_embeddings[:, idx_start:idx_end]

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if latents_all is not None and i == 0:
                        latents[:, :overlap] = (
                            latents_all[:, -overlap:]
                            + latents[:, :overlap]
                            / self.scheduler.init_noise_sigma
                            * self.scheduler.sigmas[i]
                        )

                    latent_model_input = latents 
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    ) 
                    
                    # 長度安全對齊
                    min_len = min(latent_model_input.shape[1], video_latents_current.shape[1])
                    latent_model_input = latent_model_input[:, :min_len]
                    video_latents_current_safe = video_latents_current[:, :min_len]

                    latent_model_input = torch.cat(
                        [latent_model_input, video_latents_current_safe], dim=2
                    )
                    
                    # 確保輸入不要有 NaN
                    if torch.isnan(latent_model_input).any():
                        latent_model_input = torch.nan_to_num(latent_model_input, 0.0)

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=video_embeddings_current,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    
                    if self.do_classifier_free_guidance:
                        latent_model_input = latents
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )
                        latent_model_input = torch.cat(
                            [latent_model_input, torch.zeros_like(latent_model_input)],
                            dim=2,
                        )
                        noise_pred_uncond = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=torch.zeros_like(
                                video_embeddings_current
                            ),
                            added_time_ids=added_time_ids,
                            return_dict=False,
                        )[0]

                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred - noise_pred_uncond
                        )
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )
                        latents = callback_outputs.pop("latents", latents)

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if progress_callback is not None:
                            progress_callback(i)

            if latents_all is None:
                latents_all = latents.clone()
            else:
                assert weights is not None
                latents_all[:, -overlap:] = latents[
                    :, :overlap
                ] * weights + latents_all[:, -overlap:] * (1 - weights)
                latents_all = torch.cat([latents_all, latents[:, overlap:]], dim=1)

            idx_start += stride
            torch.cuda.empty_cache()

        if not output_type == "latent":
            latents_all = latents_all.to(dtype=self.vae.dtype)

            # ================= AMD Fix: Decode Tiling =================
            try:
                self.enable_vae_tiling()
                self.vae.tile_sample_min_size = 256
                self.vae.tile_latent_min_size = 32
                print(f"AMD Fix: VAE Tiling Active for decode")
            except Exception as e:
                pass

            frames = self.decode_latents(latents_all, num_frames, decode_chunk_size)

            try:
                self.vae.disable_tiling()
            except:
                pass

            frames = self.video_processor.postprocess_video(
                video=frames, output_type=output_type
            )

        else:
            frames = latents_all

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
