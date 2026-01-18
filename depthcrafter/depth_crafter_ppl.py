from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import cv2  # <--- 修正: 確保在頂層導入 cv2

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    StableVideoDiffusionPipelineOutput,
    StableVideoDiffusionPipeline,
    retrieve_timesteps,
)
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
        chunk_size: int = 14,
    ) -> torch.Tensor:
        """
        :param video: [b, c, h, w] in range [-1, 1], the b may contain multiple videos or frames
        :param chunk_size: the chunk size to encode video
        :return: image_embeddings in shape of [b, 1024]
        """

        video_224 = _resize_with_antialiasing(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0  # [-1, 1] -> [0, 1]

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
            embeddings.append(self.image_encoder(tmp).image_embeds)  # [b, 1024]

        embeddings = torch.cat(embeddings, dim=0)  # [t, 1024]
        return embeddings

    @torch.inference_mode()
    def encode_vae_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 14,
    ):
        """
        :param video: [b, c, h, w] in range [-1, 1], the b may contain multiple videos or frames
        :param chunk_size: the chunk size to encode video
        :return: vae latents in shape of [b, c, h, w]
        """
        video_latents = []
        for i in range(0, video.shape[0], chunk_size):
            video_latents.append(
                self.vae.encode(video[i : i + chunk_size]).latent_dist.mode()
            )
        video_latents = torch.cat(video_latents, dim=0)
        return video_latents

    @staticmethod
    def check_inputs(video, height, width):
        """
        Check inputs.
        """
        if not isinstance(video, torch.Tensor) and not isinstance(video, np.ndarray):
            raise ValueError(
                f"Expected `video` to be a `torch.Tensor` or `VideoReader`, but got a {type(video)}"
            )

        # AMD Fix: 這裡的檢查可以放寬，因為我們會在後面強制 resize
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
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # ================= AMD GFX1151 FIX 1: Resize (Force Multiple of 64) =================
        def make_multiple_of_64(val):
            return ((val + 63) // 64) * 64

        original_height, original_width = height, width
        height = make_multiple_of_64(height)
        width = make_multiple_of_64(width)

        if height != original_height or width != original_width:
             print(f"AMD Fix: Resizing input from {original_width}x{original_height} to {width}x{height} (multiples of 64)")
             
             # 如果輸入是 Tensor，假設格式為 NCHW (符合 diffusers 標準)
             if isinstance(video, torch.Tensor):
                 # <--- 修正: 直接 resize，不需要 permute，避免導致花屏
                 video = _resize_with_antialiasing(video, (height, width))
                 
             # 如果輸入是 Numpy，假設格式為 NHWC
             elif isinstance(video, np.ndarray):
                 new_video = []
                 for frame in video:
                    new_video.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4))
                 video = np.array(new_video)
        # ==============================================================================

        num_frames = video.shape[0]
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 8
        if num_frames <= window_size:
            window_size = num_frames
            overlap = 0
        stride = window_size - overlap

        # 1. Check inputs
        self.check_inputs(video, height, width)

        # ================= AMD GFX1151 FIX 2: Disable Flash Attention =================
        # 這是解決 HIPBLAS_STATUS_INVALID_VALUE 的關鍵
        try:
             self.unet.set_default_attn_processor()
             self.vae.set_default_attn_processor()
             print("AMD Fix: Forced Default Attention Processor (Disabled Flash Attention)")
        except Exception as e:
             print(f"Warning: Could not set default attention processor: {e}")
        # ==============================================================================
        
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
        video = video * 2.0 - 1.0  # [0,1] -> [-1,1], in [t, c, h, w]

        if track_time:
            start_event = torch.cuda.Event(enable_timing=True)
            encode_event = torch.cuda.Event(enable_timing=True)
            denoise_event = torch.cuda.Event(enable_timing=True)
            decode_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        video_embeddings = self.encode_video(
            video, chunk_size=decode_chunk_size
        ).unsqueeze(
            0
        )  # [1, t, 1024]
        torch.cuda.empty_cache()
        # 4. Encode input image using VAE
        noise = randn_tensor(
            video.shape, generator=generator, device=device, dtype=video.dtype
        )
        video = video + noise_aug_strength * noise  # in [t, c, h, w]

        # pdb.set_trace()
        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        video_latents = self.encode_vae_video(
            video.to(self.vae.dtype),
            chunk_size=decode_chunk_size,
        ).unsqueeze(
            0
        )  # [1, t, c, h, w]

        if track_time:
            encode_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(encode_event)
            print(f"Elapsed time for encoding video: {elapsed_time_ms} ms")

        torch.cuda.empty_cache()

        # cast back to fp16 if needed
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
        )  # [1 or 2, 3]
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
        )  # [1, t, c, h, w]
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

                    latent_model_input = latents  # [1, t, c, h, w]
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )  # [1, t, c, h, w]
                    latent_model_input = torch.cat(
                        [latent_model_input, video_latents_current], dim=2
                    )
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=video_embeddings_current,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    # perform guidance
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

        if track_time:
            denoise_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = encode_event.elapsed_time(denoise_event)
            print(f"Elapsed time for denoising video: {elapsed_time_ms} ms")

        if not output_type == "latent":
            latents_all = latents_all.to(dtype=self.vae.dtype)

            # ================= AMD GFX1151 FIX 3: VAE Tiling =================
            try:
                self.vae.enable_tiling()
                
                # 安全設定
                TARGET_TILE_SIZE = 512 
                TARGET_OVERLAP = 128 
                
                if hasattr(self.vae, "tile_sample_min_size"):
                    self.vae.tile_sample_min_size = TARGET_TILE_SIZE
                
                if hasattr(self.vae, "tile_latent_min_size"): 
                    self.vae.tile_latent_min_size = int(TARGET_TILE_SIZE / 8)
                
                if hasattr(self.vae, "tile_overlap"): 
                    self.vae.tile_overlap = TARGET_OVERLAP
                    
                print(f"DepthCrafter VAE Tiling Enabled: Size={TARGET_TILE_SIZE}, Overlap={TARGET_OVERLAP}")
                
            except Exception as e:
                print(f"Warning: Failed to enable VAE tiling: {e}")

            frames = self.decode_latents(latents_all, num_frames, decode_chunk_size)

            try:
                self.vae.disable_tiling()
            except:
                pass
            # =================================================================

            if track_time:
                decode_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = denoise_event.elapsed_time(decode_event)
                print(f"Elapsed time for decoding video: {elapsed_time_ms} ms")

            frames = self.video_processor.postprocess_video(
                video=frames, output_type=output_type
            )

        else:
            frames = latents_all

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
