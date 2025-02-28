# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import unittest
from typing import Tuple, Union

import PIL.Image
import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.remote_utils import remote_decode
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    torch_device,
)
from diffusers.video_processor import VideoProcessor


enable_full_determinism()


class RemoteAutoencoderKLMixin:
    shape: Tuple[int, ...] = None
    out_hw: Tuple[int, int] = None
    endpoint: str = None
    dtype: torch.dtype = None
    scaling_factor: float = None
    shift_factor: float = None
    processor_cls: Union[VaeImageProcessor, VideoProcessor] = None

    def get_dummy_inputs(self):
        inputs = {
            "endpoint": self.endpoint,
            "tensor": torch.randn(self.shape, device=torch_device, dtype=self.dtype),
            "scaling_factor": self.scaling_factor,
            "shift_factor": self.shift_factor,
        }
        return inputs

    def test_output_type_pt(self):
        inputs = self.get_dummy_inputs()
        processor = self.processor_cls()
        output = remote_decode(output_type="pt", processor=processor, **inputs)
        self.assertTrue(isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}")
        self.assertEqual(output.height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}")
        self.assertEqual(output.width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}")

    def test_output_type_pil(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pil", **inputs)
        self.assertTrue(isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}")
        self.assertEqual(output.height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}")
        self.assertEqual(output.width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}")

    def test_output_type_pil_image_format(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pil", image_format="png", **inputs)
        self.assertTrue(isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}")
        self.assertEqual(output.height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}")
        self.assertEqual(output.width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}")
        self.assertEqual(output.format, "png", f"Expected image format `png`, got {output.format}")

    def test_output_type_pt_partial_postprocess(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", partial_postprocess=True, **inputs)
        self.assertTrue(isinstance(output, PIL.Image.Image), f"Expected `PIL.Image.Image` output, got {type(output)}")
        self.assertEqual(output.height, self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.height}")
        self.assertEqual(output.width, self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.width}")

    def test_output_type_pt_return_type_pt(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", return_type="pt", **inputs)
        self.assertTrue(isinstance(output, torch.Tensor), f"Expected `torch.Tensor` output, got {type(output)}")
        self.assertEqual(
            output.shape[2], self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.shape[2]}"
        )
        self.assertEqual(
            output.shape[3], self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.shape[3]}"
        )

    def test_output_type_pt_partial_postprocess_return_type_pt(self):
        inputs = self.get_dummy_inputs()
        output = remote_decode(output_type="pt", partial_postprocess=True, return_type="pt", **inputs)
        self.assertTrue(isinstance(output, torch.Tensor), f"Expected `torch.Tensor` output, got {type(output)}")
        self.assertEqual(
            output.shape[1], self.out_hw[0], f"Expected image height {self.out_hw[0]}, got {output.shape[1]}"
        )
        self.assertEqual(
            output.shape[2], self.out_hw[1], f"Expected image width {self.out_hw[0]}, got {output.shape[2]}"
        )

    def test_do_scaling_deprecation(self):
        inputs = self.get_dummy_inputs()
        inputs.pop("scaling_factor", None)
        inputs.pop("shift_factor", None)
        with self.assertWarns(FutureWarning) as warning:
            _ = remote_decode(output_type="pt", partial_postprocess=True, **inputs)
            self.assertEqual(
                str(warning.warnings[0].message),
                "`do_scaling` is deprecated, pass `scaling_factor` and `shift_factor` if required.",
                str(warning.warnings[0].message),
            )


class RemoteAutoencoderKLSDv1Tests(
    RemoteAutoencoderKLMixin,
    unittest.TestCase,
):
    shape = (
        1,
        4,
        64,
        64,
    )
    out_hw = (
        512,
        512,
    )
    endpoint = "https://bz0b3zkoojf30bhx.us-east-1.aws.endpoints.huggingface.cloud/"
    dtype = torch.float16
    scaling_factor = 0.18215
    shift_factor = None
    processor_cls = VaeImageProcessor
