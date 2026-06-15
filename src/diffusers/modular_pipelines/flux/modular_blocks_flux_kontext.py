# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    FluxKontextRoPEInputsStep,
    FluxPrepareLatentsStep,
    FluxRoPEInputsStep,
    FluxSetTimestepsStep,
)
from .decoders import FluxDecodeStep
from .denoise import FluxKontextDenoiseStep
from .encoders import (
    FluxKontextProcessImagesInputStep,
    FluxTextEncoderStep,
    FluxVaeEncoderStep,
)
from .inputs import (
    FluxKontextAdditionalInputsStep,
    FluxKontextSetResolutionStep,
    FluxTextInputStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Flux Kontext vae encoder (run before before_denoise)
# auto_docstring
class FluxKontextVaeEncoderStep(SequentialPipelineBlocks):
    """
    Vae encoder step that preprocess andencode the image inputs into their latent representations.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKL`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          _auto_resize (`bool`, *optional*, defaults to True):
              Whether to resize the input image to a preferred Flux Kontext resolution.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          processed_image (`None`):
              Preprocessed image tensor ready for VAE encoding.
          image_latents (`Tensor`):
              The latents representing the reference image
    """

    model_name = "flux-kontext"

    block_classes = [FluxKontextProcessImagesInputStep(), FluxVaeEncoderStep(sample_mode="argmax")]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Vae encoder step that preprocess andencode the image inputs into their latent representations."


# auto_docstring
class FluxKontextAutoVaeEncoderStep(AutoPipelineBlocks):
    """
    Vae encoder step that encode the image inputs into their latent representations.
      This is an auto pipeline block that works for image-conditioned tasks.
       - `FluxKontextVaeEncoderStep` (image_conditioned) is used when only `image` is provided. - if `image` is not
         provided, step will be skipped.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKL`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          _auto_resize (`bool`, *optional*, defaults to True):
              Whether to resize the input image to a preferred Flux Kontext resolution.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          processed_image (`None`):
              Preprocessed image tensor ready for VAE encoding.
          image_latents (`Tensor`):
              The latents representing the reference image
    """

    model_name = "flux-kontext"

    block_classes = [FluxKontextVaeEncoderStep]
    block_names = ["image_conditioned"]
    block_trigger_inputs = ["image"]

    @property
    def description(self) -> str:
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block that works for image-conditioned tasks.\n"
            + " - `FluxKontextVaeEncoderStep` (image_conditioned) is used when only `image` is provided."
            + " - if `image` is not provided, step will be skipped."
        )


# before_denoise: text2img
# auto_docstring
class FluxKontextBeforeDenoiseStep(SequentialPipelineBlocks):
    """
    Before denoise step that prepares the inputs for the denoise step for Flux Kontext

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor | NoneType`, *optional*):
              Pre-generated noisy latents for image generation.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          dtype (`dtype`, *optional*):
              The dtype of the model inputs
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          guidance_scale (`float`, *optional*, defaults to 3.5):
              Guidance scale used to prepare Flux guidance conditioning.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.

      Outputs:
          latents (`Tensor`):
              The initial latents to use for the denoising process
          timesteps (`Tensor`):
              The timesteps to use for inference
          num_inference_steps (`int`):
              The number of denoising steps to perform at inference time
          guidance (`Tensor`):
              Optional guidance to be used.
          txt_ids (`list`):
              The sequence lengths of the prompt embeds, used for RoPE calculation.
          img_ids (`list`):
              The sequence lengths of the image latents, used for RoPE calculation.
    """

    model_name = "flux-kontext"

    block_classes = [FluxPrepareLatentsStep(), FluxSetTimestepsStep(), FluxRoPEInputsStep()]
    block_names = ["prepare_latents", "set_timesteps", "prepare_rope_inputs"]

    @property
    def description(self) -> str:
        return "Before denoise step that prepares the inputs for the denoise step for Flux Kontext\n"
        "for text-to-image tasks."


# before_denoise: image-conditioned
# auto_docstring
class FluxKontextImageConditionedBeforeDenoiseStep(SequentialPipelineBlocks):
    """
    Before denoise step that prepare the inputs for the denoise step for Flux Kontext
      for image-conditioned tasks.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor | NoneType`, *optional*):
              Pre-generated noisy latents for image generation.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          dtype (`dtype`, *optional*):
              The dtype of the model inputs
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          guidance_scale (`float`, *optional*, defaults to 3.5):
              Guidance scale used to prepare Flux guidance conditioning.
          image_height (`int`, *optional*):
              Height of the encoded conditioning image.
          image_width (`int`, *optional*):
              Width of the encoded conditioning image.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.

      Outputs:
          latents (`Tensor`):
              The initial latents to use for the denoising process
          timesteps (`Tensor`):
              The timesteps to use for inference
          num_inference_steps (`int`):
              The number of denoising steps to perform at inference time
          guidance (`Tensor`):
              Optional guidance to be used.
          txt_ids (`list`):
              The sequence lengths of the prompt embeds, used for RoPE calculation.
          img_ids (`list`):
              The sequence lengths of the image latents, used for RoPE calculation.
    """

    model_name = "flux-kontext"

    block_classes = [FluxPrepareLatentsStep(), FluxSetTimestepsStep(), FluxKontextRoPEInputsStep()]
    block_names = ["prepare_latents", "set_timesteps", "prepare_rope_inputs"]

    @property
    def description(self) -> str:
        return (
            "Before denoise step that prepare the inputs for the denoise step for Flux Kontext\n"
            "for image-conditioned tasks."
        )


# auto_docstring
class FluxKontextAutoBeforeDenoiseStep(AutoPipelineBlocks):
    """
    Before denoise step that prepare the inputs for the denoise step.
      This is an auto pipeline block that works for text2image.
       - `FluxKontextBeforeDenoiseStep` (text2image) is used.
       - `FluxKontextImageConditionedBeforeDenoiseStep` (image_conditioned) is used when only `image_latents` is
         provided.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          latents (`Tensor | NoneType`, *optional*):
              Pre-generated noisy latents for image generation.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          dtype (`dtype`, *optional*):
              The dtype of the model inputs
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          guidance_scale (`float`, *optional*, defaults to 3.5):
              Guidance scale used to prepare Flux guidance conditioning.
          image_height (`int`, *optional*):
              Height of the encoded conditioning image.
          image_width (`int`, *optional*):
              Width of the encoded conditioning image.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.

      Outputs:
          latents (`Tensor`):
              The initial latents to use for the denoising process
          timesteps (`Tensor`):
              The timesteps to use for inference
          num_inference_steps (`int`):
              The number of denoising steps to perform at inference time
          guidance (`Tensor`):
              Optional guidance to be used.
          txt_ids (`list`):
              The sequence lengths of the prompt embeds, used for RoPE calculation.
          img_ids (`list`):
              The sequence lengths of the image latents, used for RoPE calculation.
    """

    model_name = "flux-kontext"

    block_classes = [FluxKontextImageConditionedBeforeDenoiseStep, FluxKontextBeforeDenoiseStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self) -> str:
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is an auto pipeline block that works for text2image.\n"
            + " - `FluxKontextBeforeDenoiseStep` (text2image) is used.\n"
            + " - `FluxKontextImageConditionedBeforeDenoiseStep` (image_conditioned) is used when only `image_latents` is provided.\n"
        )


# inputs: Flux Kontext
# auto_docstring
class FluxKontextInputStep(SequentialPipelineBlocks):
    """
    Input step that prepares the inputs for the both text2img and img2img denoising step. It:
       - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents`).
       - update height/width based `image_latents`, patchify `image_latents`.

      Inputs:
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          max_area (`int`, *optional*, defaults to 1048576):
              Maximum pixel area to fit the generated resolution within while preserving aspect ratio.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from the text encoder step.
          pooled_prompt_embeds (`Tensor`, *optional*):
              Pre-generated pooled text embeddings. Can be generated from text_encoder step.
          image_latents (`Tensor | NoneType`, *optional*):
              Optional latent input `image_latents` to patchify and batch-expand.

      Outputs:
          height (`int`):
              The height of the initial noisy latents
          width (`int`):
              The width of the initial noisy latents
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt
          dtype (`dtype`):
              Data type of model tensor inputs (determined by `prompt_embeds`)
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation
          pooled_prompt_embeds (`Tensor`):
              pooled text embeddings used to guide the image generation
          image_height (`int`):
              The height of the image latents
          image_width (`int`):
              The width of the image latents
    """

    model_name = "flux-kontext"
    block_classes = [FluxKontextSetResolutionStep(), FluxTextInputStep(), FluxKontextAdditionalInputsStep()]
    block_names = ["set_resolution", "text_inputs", "additional_inputs"]

    @property
    def description(self) -> str:
        return (
            "Input step that prepares the inputs for the both text2img and img2img denoising step. It:\n"
            " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents`).\n"
            " - update height/width based `image_latents`, patchify `image_latents`."
        )


# auto_docstring
class FluxKontextAutoInputStep(AutoPipelineBlocks):
    """
    Input step that standardize the inputs for the denoising step, e.g. make sure inputs have consistent batch size,
    and patchified.
       This is an auto pipeline block that works for text2image/img2img tasks.
       - `FluxKontextInputStep` (image_conditioned) is used when `image_latents` is provided.
       - `FluxKontextInputStep` is also capable of handling text2image task when `image_latent` isn't present.

      Inputs:
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          max_area (`int`, *optional*, defaults to 1048576):
              Maximum pixel area to fit the generated resolution within while preserving aspect ratio.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from the text encoder step.
          pooled_prompt_embeds (`Tensor`, *optional*):
              Pre-generated pooled text embeddings. Can be generated from text_encoder step.
          image_latents (`Tensor | NoneType`, *optional*):
              Optional latent input `image_latents` to patchify and batch-expand.

      Outputs:
          height (`int`):
              The height of the initial noisy latents
          width (`int`):
              The width of the initial noisy latents
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt
          dtype (`dtype`):
              Data type of model tensor inputs (determined by `prompt_embeds`)
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation
          pooled_prompt_embeds (`Tensor`):
              pooled text embeddings used to guide the image generation
          image_height (`int`):
              The height of the image latents
          image_width (`int`):
              The width of the image latents
    """

    model_name = "flux-kontext"
    block_classes = [FluxKontextInputStep, FluxTextInputStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self) -> str:
        return (
            "Input step that standardize the inputs for the denoising step, e.g. make sure inputs have consistent batch size, and patchified. \n"
            " This is an auto pipeline block that works for text2image/img2img tasks.\n"
            + " - `FluxKontextInputStep` (image_conditioned) is used when `image_latents` is provided.\n"
            + " - `FluxKontextInputStep` is also capable of handling text2image task when `image_latent` isn't present."
        )


# auto_docstring
class FluxKontextCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core step that performs the denoising process for Flux Kontext.
      This step supports text-to-image and image-conditioned tasks for Flux Kontext:
       - for image-conditioned generation, you need to provide `image_latents`
       - for text-to-image generation, all you need to provide is prompt embeddings.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`FluxTransformer2DModel`)

      Inputs:
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          max_area (`int`, *optional*, defaults to 1048576):
              Maximum pixel area to fit the generated resolution within while preserving aspect ratio.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from the text encoder step.
          pooled_prompt_embeds (`Tensor`, *optional*):
              Pre-generated pooled text embeddings. Can be generated from text_encoder step.
          image_latents (`Tensor | NoneType`, *optional*):
              Optional latent input `image_latents` to patchify and batch-expand.
          latents (`Tensor | NoneType`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          guidance_scale (`float`, *optional*, defaults to 3.5):
              Guidance scale used to prepare Flux guidance conditioning.
          joint_attention_kwargs (`dict | NoneType`, *optional*):
              Additional kwargs forwarded to the transformer's attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux-kontext"
    block_classes = [FluxKontextAutoInputStep, FluxKontextAutoBeforeDenoiseStep, FluxKontextDenoiseStep]
    block_names = ["input", "before_denoise", "denoise"]

    @property
    def description(self) -> str:
        return (
            "Core step that performs the denoising process for Flux Kontext.\n"
            + "This step supports text-to-image and image-conditioned tasks for Flux Kontext:\n"
            + " - for image-conditioned generation, you need to provide `image_latents`\n"
            + " - for text-to-image generation, all you need to provide is prompt embeddings."
        )

    @property
    def outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("latents"),
        ]


AUTO_BLOCKS_KONTEXT = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep()),
        ("vae_encoder", FluxKontextAutoVaeEncoderStep()),
        ("denoise", FluxKontextCoreDenoiseStep()),
        ("decode", FluxDecodeStep()),
    ]
)


# auto_docstring
class FluxKontextAutoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline for image-to-image using Flux Kontext.

      Supported workflows:
        - `image_conditioned`: requires `image`, `prompt`
        - `text2image`: requires `prompt`

      Components:
          text_encoder (`CLIPTextModel`) tokenizer (`CLIPTokenizer`) text_encoder_2 (`T5EncoderModel`) tokenizer_2
          (`T5TokenizerFast`) image_processor (`VaeImageProcessor`) vae (`AutoencoderKL`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) transformer (`FluxTransformer2DModel`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          prompt_2 (`str | list[str]`, *optional*):
              Optional second prompt or prompts for the T5 text encoder.
          max_sequence_length (`int`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.
          joint_attention_kwargs (`None`, *optional*):
              Additional kwargs forwarded to the transformer's attention processors.
          image (`PIL.Image.Image | list[PIL.Image.Image]`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          _auto_resize (`bool`, *optional*, defaults to True):
              Whether to resize the input image to a preferred Flux Kontext resolution.
          generator (`torch.Generator`, *optional*):
              Torch generator for deterministic generation.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          max_area (`int`, *optional*, defaults to 1048576):
              Maximum pixel area to fit the generated resolution within while preserving aspect ratio.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          image_latents (`Tensor`, *optional*):
              Optional latent input `image_latents` to patchify and batch-expand.
          latents (`Tensor | NoneType`, *optional*):
              Pre-generated noisy latents for image generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list[float]`, *optional*):
              Custom sigmas for the denoising process.
          guidance_scale (`float`, *optional*, defaults to 3.5):
              Guidance scale used to prepare Flux guidance conditioning.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "flux-kontext"

    block_classes = AUTO_BLOCKS_KONTEXT.values()
    block_names = AUTO_BLOCKS_KONTEXT.keys()
    _workflow_map = {
        "image_conditioned": {"image": True, "prompt": True},
        "text2image": {"prompt": True},
    }

    @property
    def description(self) -> str:
        return "Modular pipeline for image-to-image using Flux Kontext."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]
