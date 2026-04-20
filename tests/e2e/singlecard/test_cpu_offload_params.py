# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""E2E test for cpu-offload-params with prefetch backend on NPU.

Tests both eager mode and ACL Graph mode.
"""

import pytest
from vllm import LLM, SamplingParams
from vllm.config import OffloadConfig


@pytest.fixture
def prompts():
    return [
        "The theory of relativity was proposed by",
        "Hello, how are you today?",
        "Explain quantum computing in simple terms.",
    ]


class TestCPUPrefetchOffloadParams:
    """E2E tests for cpu-offload-params with prefetch backend."""

    @pytest.mark.parametrize(
        "group_size,num_in_group,prefetch_step",
        [
            (4, 1, 1),
            (8, 2, 1),
            (4, 1, 2),
        ],
    )
    def test_prefetch_offload_eager(self, prompts, group_size, num_in_group, prefetch_step):
        """Test basic generation with prefetch offloading in eager mode."""
        llm = LLM(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            offload_group_size=group_size,
            offload_num_in_group=num_in_group,
            offload_prefetch_step=prefetch_step,
        )
        sampling_params = SamplingParams(max_tokens=20, temperature=0.0)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        assert len(outputs) == len(prompts)
        for output in outputs:
            assert len(output.outputs) == 1
            assert len(output.outputs[0].token_ids) > 0

        del llm

    @pytest.mark.parametrize(
        "group_size,num_in_group,prefetch_step",
        [
            (4, 1, 1),
            (8, 2, 1),
        ],
    )
    def test_prefetch_offload_acl_graph(self, prompts, group_size, num_in_group, prefetch_step):
        """Test prefetch offloading with ACL Graph enabled (default mode)."""
        llm = LLM(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            gpu_memory_utilization=0.8,
            # enforce_eager not set -> ACL Graph (PIECEWISE) enabled by default
            offload_group_size=group_size,
            offload_num_in_group=num_in_group,
            offload_prefetch_step=prefetch_step,
        )
        sampling_params = SamplingParams(max_tokens=20, temperature=0.0)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        assert len(outputs) == len(prompts)
        for output in outputs:
            assert len(output.outputs) == 1
            assert len(output.outputs[0].token_ids) > 0

        del llm

    def test_prefetch_offload_correctness_eager(self, prompts):
        """Compare output with and without offloading in eager mode."""
        sampling_params = SamplingParams(max_tokens=20, temperature=0.0)

        # Baseline: no offloading
        llm_baseline = LLM(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
        )
        outputs_baseline = llm_baseline.generate(prompts, sampling_params, use_tqdm=False)
        del llm_baseline

        # With prefetch offloading
        llm_offload = LLM(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            offload_group_size=4,
            offload_num_in_group=1,
            offload_prefetch_step=1,
        )
        outputs_offload = llm_offload.generate(prompts, sampling_params, use_tqdm=False)
        del llm_offload

        # Compare outputs: with temperature=0.0, outputs should be identical
        for i, (base, offload) in enumerate(zip(outputs_baseline, outputs_offload)):
            base_text = base.outputs[0].text
            offload_text = offload.outputs[0].text
            assert base_text == offload_text, (
                f"Prompt {i}: outputs differ.\n"
                f"Baseline: {base_text!r}\n"
                f"Offload:  {offload_text!r}"
            )

    def test_prefetch_offload_correctness_graph(self, prompts):
        """Compare output with and without offloading in ACL Graph mode."""
        sampling_params = SamplingParams(max_tokens=20, temperature=0.0)

        # Baseline: no offloading, ACL Graph enabled
        llm_baseline = LLM(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            gpu_memory_utilization=0.8,
        )
        outputs_baseline = llm_baseline.generate(prompts, sampling_params, use_tqdm=False)
        del llm_baseline

        # With prefetch offloading, ACL Graph enabled
        llm_offload = LLM(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            gpu_memory_utilization=0.8,
            offload_group_size=4,
            offload_num_in_group=1,
            offload_prefetch_step=1,
        )
        outputs_offload = llm_offload.generate(prompts, sampling_params, use_tqdm=False)
        del llm_offload

        for i, (base, offload) in enumerate(zip(outputs_baseline, outputs_offload)):
            base_text = base.outputs[0].text
            offload_text = offload.outputs[0].text
            assert base_text == offload_text, (
                f"Prompt {i}: outputs differ in ACL Graph mode.\n"
                f"Baseline: {base_text!r}\n"
                f"Offload:  {offload_text!r}"
            )

    def test_uva_backend_graceful_fallback(self):
        """UVA backend should gracefully fall back to NoopOffloader on NPU."""
        llm = LLM(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            offload_backend="uva",
            cpu_offload_gb=10,
        )
        sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

        # Should still work (falls back to NoopOffloader)
        outputs = llm.generate("Hello world", sampling_params, use_tqdm=False)
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) > 0

        del llm
