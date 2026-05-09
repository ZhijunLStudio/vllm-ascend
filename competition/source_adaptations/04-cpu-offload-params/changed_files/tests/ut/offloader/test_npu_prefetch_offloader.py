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
"""Unit tests for NPUPrefetchOffloader.

Requires NPU hardware for torch.npu.Stream/Event and module placement.
Run with: VLLM_VERSION=0.19.0 pytest tests/ut/offloader/test_npu_prefetch_offloader.py -v
"""

import torch
import torch.nn as nn
import pytest

from vllm.model_executor.offloader.base import NoopOffloader, create_offloader
from vllm.model_executor.offloader.prefetch import ParamInfo, StaticBufferPool

from vllm_ascend.offloader.npu_prefetch import NPUPrefetchOffloader, _NPUModuleOffloader

NPU_DEVICE = torch.device("npu:0")


class SimpleMLP(nn.Module):
    """Simple MLP for testing offloader."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_up_proj(x))


def make_npu_model(num_layers=4, hidden_size=64):
    """Create a model with layers on NPU device."""
    layers = nn.ModuleList([SimpleMLP(hidden_size).to(NPU_DEVICE) for _ in range(num_layers)])
    return layers


class TestNPUPrefetchOffloaderInit:
    """Test NPUPrefetchOffloader initialization."""

    def test_creates_npu_stream(self):
        """Verify the offloader creates NPU stream, not CUDA stream."""
        offloader = NPUPrefetchOffloader(
            group_size=2, num_in_group=1, prefetch_step=1
        )
        assert offloader.copy_stream is not None
        assert offloader.group_size == 2
        assert offloader.num_in_group == 1
        assert offloader.prefetch_step == 1
        assert offloader.module_offloaders == []
        assert offloader.total_offloaded_bytes == 0

    def test_default_params(self):
        """Test default parameter values."""
        offloader = NPUPrefetchOffloader(
            group_size=4, num_in_group=1, prefetch_step=1
        )
        assert offloader.offload_params == set()
        assert offloader.mode == "cpu"

    def test_offload_params(self):
        """Test offload_params parameter."""
        params = {"gate_up_proj", "down_proj"}
        offloader = NPUPrefetchOffloader(
            group_size=4, num_in_group=1, prefetch_step=1,
            offload_params=params
        )
        assert offloader.offload_params == params


class TestNPUPrefetchOffloaderWrapModules:
    """Test wrap_modules layer selection logic."""

    def test_layer_selection_group_4_num_1(self):
        """With group_size=4, num_in_group=1: only layer 3 is offloaded."""
        layers = make_npu_model(4)
        offloader = NPUPrefetchOffloader(
            group_size=4, num_in_group=1, prefetch_step=1
        )
        result = offloader.wrap_modules(iter(list(layers)))
        assert len(offloader.module_offloaders) == 1
        assert offloader.module_offloaders[0].layer_idx == 0

    def test_layer_selection_group_2_num_1(self):
        """With group_size=2, num_in_group=1: layers 1, 3 are offloaded."""
        layers = make_npu_model(4)
        offloader = NPUPrefetchOffloader(
            group_size=2, num_in_group=1, prefetch_step=1
        )
        result = offloader.wrap_modules(iter(list(layers)))
        assert len(offloader.module_offloaders) == 2

    def test_layer_selection_group_4_num_2(self):
        """With group_size=4, num_in_group=2: layers 2, 3 are offloaded."""
        layers = make_npu_model(4)
        offloader = NPUPrefetchOffloader(
            group_size=4, num_in_group=2, prefetch_step=1
        )
        result = offloader.wrap_modules(iter(list(layers)))
        assert len(offloader.module_offloaders) == 2

    def test_offload_params_whitelist(self):
        """Test selective parameter offloading with whitelist.

        Uses 'gate_up_proj' as filter which matches 'gate_up_proj.weight'
        but not 'down_proj.weight' (segment matching).
        """
        layers = make_npu_model(4)
        offloader = NPUPrefetchOffloader(
            group_size=4, num_in_group=1, prefetch_step=1,
            offload_params={"gate_up_proj"}
        )
        result = offloader.wrap_modules(iter(list(layers)))

        assert len(offloader.module_offloaders) == 1
        module_offloader = offloader.module_offloaders[0]
        param_names = list(module_offloader._param_offloaders.keys())
        assert any("gate_up_proj" in name for name in param_names)
        assert not any("down_proj" in name for name in param_names)

    def test_returns_all_modules(self):
        """wrap_modules should return all modules, not just offloaded ones."""
        layers = make_npu_model(4)
        offloader = NPUPrefetchOffloader(
            group_size=2, num_in_group=1, prefetch_step=1
        )
        result = offloader.wrap_modules(iter(list(layers)))
        assert len(result) == 4

    def test_wrap_modules_called_twice_raises(self):
        """Calling wrap_modules twice should raise AssertionError."""
        layers = make_npu_model(4)
        offloader = NPUPrefetchOffloader(
            group_size=4, num_in_group=1, prefetch_step=1
        )
        offloader.wrap_modules(iter(list(layers)))
        with pytest.raises(AssertionError, match="wrap_modules should only be called once"):
            offloader.wrap_modules(iter(list(layers)))


class TestNPUPrefetchOffloaderSyncMethods:
    """Test sync/join methods."""

    def test_sync_prev_onload_no_modules(self):
        """sync_prev_onload should not crash when no modules are offloaded."""
        offloader = NPUPrefetchOffloader(
            group_size=2, num_in_group=1, prefetch_step=1
        )
        offloader.sync_prev_onload()

    def test_join_after_forward_no_modules(self):
        """join_after_forward should not crash when no modules are offloaded."""
        offloader = NPUPrefetchOffloader(
            group_size=2, num_in_group=1, prefetch_step=1
        )
        offloader.join_after_forward()

    def test_post_init_no_modules(self):
        """post_init should not crash when no modules are offloaded."""
        offloader = NPUPrefetchOffloader(
            group_size=2, num_in_group=1, prefetch_step=1
        )
        offloader.post_init()


class TestCreateOffloaderPatch:
    """Test that the factory patch works correctly."""

    def test_prefetch_returns_npu_offloader(self):
        """prefetch backend should return NPUPrefetchOffloader."""
        from vllm.config import OffloadConfig
        config = OffloadConfig(
            offload_backend="prefetch",
            prefetch={"offload_group_size": 4, "offload_num_in_group": 1, "offload_prefetch_step": 1}
        )
        offloader = create_offloader(config)
        assert isinstance(offloader, NPUPrefetchOffloader)

    def test_uva_returns_noop(self):
        """UVA backend should return NoopOffloader on NPU."""
        from vllm.config import OffloadConfig
        config = OffloadConfig(
            offload_backend="uva",
            uva={"cpu_offload_gb": 10}
        )
        offloader = create_offloader(config)
        assert isinstance(offloader, NoopOffloader)

    def test_auto_prefetch_returns_npu_offloader(self):
        """auto backend with offload_group_size > 0 should return NPUPrefetchOffloader."""
        from vllm.config import OffloadConfig
        config = OffloadConfig(
            offload_backend="auto",
            prefetch={"offload_group_size": 4, "offload_num_in_group": 1, "offload_prefetch_step": 1}
        )
        offloader = create_offloader(config)
        assert isinstance(offloader, NPUPrefetchOffloader)

    def test_auto_uva_returns_noop(self):
        """auto backend with cpu_offload_gb > 0 should return NoopOffloader on NPU."""
        from vllm.config import OffloadConfig
        config = OffloadConfig(
            offload_backend="auto",
            uva={"cpu_offload_gb": 10}
        )
        offloader = create_offloader(config)
        assert isinstance(offloader, NoopOffloader)

    def test_empty_returns_noop(self):
        """Empty config should return NoopOffloader."""
        from vllm.config import OffloadConfig
        config = OffloadConfig()
        offloader = create_offloader(config)
        assert isinstance(offloader, NoopOffloader)


class TestStaticBufferPoolNPU:
    """Test StaticBufferPool allocation on NPU."""

    def test_allocate_on_npu(self):
        """Verify buffer pool allocates on NPU device."""
        param_infos = [
            ParamInfo(name="weight", shape=(128, 64), stride=(64, 1), dtype=torch.float16),
            ParamInfo(name="bias", shape=(64,), stride=(1,), dtype=torch.float16),
        ]
        pool = StaticBufferPool(
            param_infos=param_infos,
            slot_capacity=2,
            device=NPU_DEVICE,
        )
        buf = pool.get_buffer("weight", (128, 64), (64, 1), torch.float16, slot_idx=0)
        assert buf.device.type == "npu"
        assert buf.shape == (128, 64)
        assert buf.dtype == torch.float16

    def test_buffer_slot_reuse(self):
        """Verify slots are reused circularly."""
        param_infos = [
            ParamInfo(name="weight", shape=(32, 32), stride=(32, 1), dtype=torch.float32),
        ]
        pool = StaticBufferPool(
            param_infos=param_infos,
            slot_capacity=2,
            device=NPU_DEVICE,
        )
        buf0 = pool.get_buffer("weight", (32, 32), (32, 1), torch.float32, slot_idx=0)
        buf1 = pool.get_buffer("weight", (32, 32), (32, 1), torch.float32, slot_idx=1)
        buf2 = pool.get_buffer("weight", (32, 32), (32, 1), torch.float32, slot_idx=2)
        # slot_idx=2 should wrap to slot 0
        assert buf2.data_ptr() == buf0.data_ptr()


class TestNPUModuleOffloader:
    """Test _NPUModuleOffloader creation."""

    def test_creates_npu_event(self):
        """Verify _NPUModuleOffloader creates NPU event, not CUDA event."""
        model = SimpleMLP().to(NPU_DEVICE)
        copy_stream = torch.npu.Stream()
        offloader = _NPUModuleOffloader(
            mode="cpu",
            module=model,
            copy_stream=copy_stream,
            whitelist_param_names=["gate_up_proj.weight", "down_proj.weight"],
            layer_idx=0,
        )
        assert offloader._copy_done_event is not None
        assert offloader._prefetch_in_capture is False
        assert offloader._event_valid_for_eager is False

    def test_param_offloaders_created(self):
        """Verify param offloaders are created for whitelisted params."""
        model = SimpleMLP().to(NPU_DEVICE)
        copy_stream = torch.npu.Stream()
        offloader = _NPUModuleOffloader(
            mode="cpu",
            module=model,
            copy_stream=copy_stream,
            whitelist_param_names=["gate_up_proj.weight"],
            layer_idx=0,
        )
        assert "gate_up_proj.weight" in offloader._param_offloaders
        assert "down_proj.weight" not in offloader._param_offloaders

    def test_cpu_param_offloader_creates_cpu_storage(self):
        """Verify _CpuParamOffloader moves params to CPU."""
        model = SimpleMLP().to(NPU_DEVICE)
        copy_stream = torch.npu.Stream()
        offloader = _NPUModuleOffloader(
            mode="cpu",
            module=model,
            copy_stream=copy_stream,
            whitelist_param_names=["gate_up_proj.weight"],
            layer_idx=0,
        )
        param_offloader = offloader._param_offloaders["gate_up_proj.weight"]
        assert param_offloader._cpu_storage is not None
        assert param_offloader._cpu_storage.device.type == "cpu"
        assert param_offloader.offloaded_bytes > 0
