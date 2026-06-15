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

import torch

from diffusers import Kandinsky5Transformer3DModel

from ...testing_utils import torch_device


class Kandinsky5Transformer3DModelTests(unittest.TestCase):
    def get_model(self):
        return Kandinsky5Transformer3DModel(
            in_visual_dim=4,
            in_text_dim=8,
            in_text_dim2=6,
            time_dim=8,
            out_visual_dim=4,
            patch_size=(1, 1, 1),
            model_dim=8,
            ff_dim=16,
            num_text_blocks=1,
            num_visual_blocks=1,
            axes_dims=(2, 2, 4),
            attention_type="regular",
        ).to(torch_device)

    def get_inputs(self):
        return {
            "hidden_states": torch.randn(1, 1, 2, 2, 4, device=torch_device),
            "encoder_hidden_states": torch.randn(1, 3, 8, device=torch_device),
            "timestep": torch.tensor([1.0], device=torch_device),
            "pooled_projections": torch.randn(1, 6, device=torch_device),
            "visual_rope_pos": [
                torch.arange(1, device=torch_device),
                torch.arange(2, device=torch_device),
                torch.arange(2, device=torch_device),
            ],
            "text_rope_pos": torch.arange(3, device=torch_device),
        }

    def test_return_dict_false_returns_tuple(self):
        model = self.get_model()
        output = model(**self.get_inputs(), return_dict=False)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].shape, (1, 1, 2, 2, 4))

    def test_no_split_modules_are_declared(self):
        self.assertEqual(
            Kandinsky5Transformer3DModel._no_split_modules,
            ["Kandinsky5TransformerEncoderBlock", "Kandinsky5TransformerDecoderBlock"],
        )
