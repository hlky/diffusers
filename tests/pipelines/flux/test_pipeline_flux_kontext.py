import gc
import unittest

import numpy as np
import PIL.Image
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FasterCacheConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)

from ...testing_utils import backend_empty_cache, require_big_accelerator, slow, torch_device
from ..test_pipelines_common import (
    FasterCacheTesterMixin,
    FluxIPAdapterTesterMixin,
    PipelineTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
)


class FluxKontextPipelineFastTests(
    unittest.TestCase,
    PipelineTesterMixin,
    FluxIPAdapterTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
    FasterCacheTesterMixin,
):
    pipeline_class = FluxKontextPipeline
    params = frozenset(
        ["image", "prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_params = frozenset(["image", "prompt"])

    # there is no xformers processor for Flux
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    faster_cache_config = FasterCacheConfig(
        spatial_attention_block_skip_range=2,
        spatial_attention_timestep_skip_range=(-1, 901),
        unconditional_batch_skip_range=2,
        attention_weight_callback=lambda _: 0.5,
        is_guidance_distilled=True,
    )

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_2 = T5EncoderModel(config)

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "vae": vae,
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        image = PIL.Image.new("RGB", (32, 32), 0)
        inputs = {
            "image": image,
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_area": 8 * 8,
            "max_sequence_length": 48,
            "output_type": "np",
            "_auto_resize": False,
        }
        return inputs

    def test_flux_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        # For some reasons, they don't show large differences
        assert max_diff > 1e-6

    def test_flux_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        height_width_pairs = [(32, 32), (72, 57)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width, "max_area": height * width})
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)

    def test_flux_true_cfg(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)
        inputs.pop("generator")

        no_true_cfg_out = pipe(**inputs, generator=torch.manual_seed(0)).images[0]
        inputs["negative_prompt"] = "bad quality"
        inputs["true_cfg_scale"] = 2.0
        true_cfg_out = pipe(**inputs, generator=torch.manual_seed(0)).images[0]
        assert not np.allclose(no_true_cfg_out, true_cfg_out)

    def test_negative_prompt_embeds_shape_validation(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        prompt_embeds = torch.zeros((1, 48, 32), device=torch_device)
        pooled_prompt_embeds = torch.zeros((1, 32), device=torch_device)
        negative_prompt_embeds = torch.zeros((1, 47, 32), device=torch_device)
        negative_pooled_prompt_embeds = torch.zeros((1, 32), device=torch_device)

        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            pipe.check_inputs(
                prompt=None,
                prompt_2=None,
                height=8,
                width=8,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )


@slow
@require_big_accelerator
class FluxKontextPipelineSlowTests(unittest.TestCase):
    pipeline_class = FluxKontextPipeline
    repo_id = "black-forest-labs/FLUX.1-Kontext-dev"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        prompt_embeds = torch.load(
            hf_hub_download(repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/prompt_embeds.pt")
        ).to(torch_device)
        pooled_prompt_embeds = torch.load(
            hf_hub_download(
                repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/pooled_prompt_embeds.pt"
            )
        ).to(torch_device)
        return {
            "image": PIL.Image.new("RGB", (512, 512), 0),
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "max_sequence_length": 256,
            "height": 512,
            "width": 512,
            "max_area": 512 * 512,
            "_auto_resize": False,
            "output_type": "np",
            "generator": generator,
        }

    def test_flux_kontext_inference(self):
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id, torch_dtype=torch.bfloat16, text_encoder=None, text_encoder_2=None
        ).to(torch_device)

        image = pipe(**self.get_inputs()).images[0]

        self.assertEqual(image.shape, (512, 512, 3))
        self.assertTrue(np.isfinite(image).all())
        self.assertGreater(float(np.abs(image).sum()), 0.0)
