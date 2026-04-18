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
"""NPU-specific prefetch-based CPU offloading.

This is the NPU port of vllm's PrefetchOffloader, replacing all
torch.cuda.* calls with torch.npu.* equivalents.

Shared building blocks (ParamInfo, StaticBufferPool, _CpuParamOffloader)
are imported directly from upstream since they contain no CUDA-specific code.
"""

import sys
import traceback
from collections.abc import Generator
from os import getpid

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.model_executor.offloader.prefetch import (
    ParamInfo,
    StaticBufferPool,
    _CpuParamOffloader,
)
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

DBG = False


def _dbg(msg):
    if DBG:
        print(f"[NPU_OFFLOAD pid={getpid()}] {msg}", file=sys.stderr, flush=True)


class NPUPrefetchOffloader(BaseOffloader):
    """NPU-specific prefetch offloader using torch.npu APIs."""

    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        offload_params: set[str] | None = None,
        mode: str = "cpu",
    ):
        self.group_size = group_size
        self.num_in_group = num_in_group
        self.prefetch_step = prefetch_step
        self.offload_params = offload_params or set()
        self.mode = mode
        self.copy_stream = torch.npu.Stream()
        self.module_offloaders: list[_NPUModuleOffloader] = []
        self.buffer_pool: StaticBufferPool | None = None
        self.total_offloaded_bytes = 0
        _dbg(f"NPUPrefetchOffloader created: group={group_size}, num={num_in_group}, step={prefetch_step}")

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        assert len(self.module_offloaders) == 0, "wrap_modules should only be called once"

        all_modules = []
        offload_modules = []

        for module_index, module in enumerate(modules_generator):
            all_modules.append(module)
            if module_index % self.group_size >= self.group_size - self.num_in_group:
                if self.offload_params:
                    whitelist = [
                        name for name, _ in module.named_parameters()
                        if any(f".{p}." in f".{name}." for p in self.offload_params)
                    ]
                else:
                    whitelist = [name for name, _ in module.named_parameters()]
                if not whitelist:
                    continue
                offload_modules.append(module)
                self.module_offloaders.append(
                    _NPUModuleOffloader(
                        mode=self.mode, module=module,
                        copy_stream=self.copy_stream,
                        whitelist_param_names=whitelist,
                        layer_idx=len(self.module_offloaders),
                    )
                )

        _dbg(f"wrap_modules: {len(all_modules)} total, {len(offload_modules)} offloaded")
        for index, module in enumerate(offload_modules):
            self._hook_module_forward(index, module)
        return all_modules

    def _hook_module_forward(self, index: int, module: nn.Module):
        original_forward = module.forward
        prefetch_step = self.prefetch_step
        offloaders_len = lambda: len(self.module_offloaders)

        def forward(*args, **kwargs):
            module.forward = original_forward
            input_tensor = args[0] if args else kwargs.get("hidden_states")
            torch.ops.vllm.wait_prefetch(input_tensor, index)
            output = original_forward(*args, **kwargs)
            next_index = (index + prefetch_step) % offloaders_len()
            if isinstance(output, tuple):
                torch.ops.vllm.start_prefetch(output[0], next_index)
            else:
                torch.ops.vllm.start_prefetch(output, next_index)
            module.forward = forward
            return output

        module.forward = forward

    def _wait_for_layer(self, layer_idx: int):
        offloader = self.module_offloaders[layer_idx]
        if torch.npu.is_current_stream_capturing():
            if not offloader._prefetch_in_capture:
                return
            torch.npu.current_stream().wait_event(offloader._copy_done_event)
            offloader._prefetch_in_capture = False
        else:
            if offloader._event_valid_for_eager:
                torch.npu.current_stream().wait_event(offloader._copy_done_event)
            else:
                torch.npu.current_stream().wait_stream(self.copy_stream)

    def sync_prev_onload(self):
        torch.npu.current_stream().wait_stream(self.copy_stream)

    def _start_prefetch(self, layer_idx: int):
        offloader = self.module_offloaders[layer_idx]
        offloader.start_onload_to_static()

    def join_after_forward(self):
        if not self.module_offloaders:
            return
        for offloader in self.module_offloaders:
            if offloader._prefetch_in_capture:
                torch.npu.current_stream().wait_event(offloader._copy_done_event)
                offloader._prefetch_in_capture = False

    def post_init(self):
        _dbg(f"post_init ENTER: {len(self.module_offloaders)} offloaders")
        try:
            for offloader in self.module_offloaders:
                offloader.sync_cpu_storage()

            param_infos: list[ParamInfo] = []
            device: torch.device | None = None
            for offloader in self.module_offloaders:
                param_infos.extend(offloader.get_param_infos())
                if device is None:
                    device = offloader.device

            if device is None:
                _dbg("post_init: no device, returning")
                return

            _dbg(f"post_init: allocating buffer pool on {device}, {len(param_infos)} params")
            self.buffer_pool = StaticBufferPool(
                param_infos=param_infos,
                slot_capacity=self.prefetch_step,
                device=device,
            )

            for idx, offloader in enumerate(self.module_offloaders):
                slot_idx = idx % self.prefetch_step
                offloader.assign_buffer_slot(self.buffer_pool, slot_idx)
                for name, po in offloader._param_offloaders.items():
                    _dbg(f"  param {name}: device={po._param.data.device}")

            for offloader in self.module_offloaders:
                offloader.post_init()
                self.total_offloaded_bytes += offloader.offloaded_bytes

            _dbg(f"post_init done: {self.total_offloaded_bytes / 1e9:.4f} GB offloaded, "
                 f"{self.buffer_pool.total_bytes / 1e9:.4f} GB buffer pool")

            for i in range(min(self.prefetch_step, len(self.module_offloaders))):
                self.module_offloaders[i].start_onload_to_static()
        except Exception as e:
            _dbg(f"post_init EXCEPTION: {e}")
            traceback.print_exc(file=sys.stderr)
            raise


class _NPUModuleOffloader:

    def __init__(self, mode, module, copy_stream, whitelist_param_names, layer_idx):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.copy_stream = copy_stream
        self.layer_idx = layer_idx
        self.offloaded_bytes = 0
        self._copy_done_event = torch.npu.Event()
        self._event_valid_for_eager = False
        self._prefetch_in_capture = False
        self._buffer_pool: StaticBufferPool | None = None
        self._buffer_slot_idx: int = 0

        assert self.device != torch.device("cpu"), "Module parameters should be on NPU"

        param_dict = dict(self.module.named_parameters())
        assert all(name in param_dict for name in whitelist_param_names), (
            f"Whitelist params {whitelist_param_names} not found in module params"
        )
        self._param_offloaders = {
            name: _CpuParamOffloader(module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):
        for param_offloader in self._param_offloaders.values():
            param_offloader.post_init()
            self.offloaded_bytes += param_offloader.offloaded_bytes

    def sync_cpu_storage(self):
        for param_offloader in self._param_offloaders.values():
            param_offloader.sync_cpu_storage()
        deleted = [name for name, o in self._param_offloaders.items()
                   if getattr(o, "_param_deleted", False)]
        if deleted:
            for name in deleted:
                del self._param_offloaders[name]

    def get_param_infos(self) -> list[ParamInfo]:
        infos = []
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            assert cpu_storage is not None
            infos.append(ParamInfo(
                name=name, shape=tuple(cpu_storage.shape),
                stride=tuple(cpu_storage.stride()), dtype=cpu_storage.dtype,
            ))
        return infos

    def assign_buffer_slot(self, pool: StaticBufferPool, slot_idx: int):
        self._buffer_pool = pool
        self._buffer_slot_idx = slot_idx
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            assert cpu_storage is not None
            buffer = pool.get_buffer(
                name=name, shape=tuple(cpu_storage.shape),
                stride=tuple(cpu_storage.stride()), dtype=cpu_storage.dtype,
                slot_idx=slot_idx,
            )
            offloader.assign_static_buffer(buffer)

    def start_onload_to_static(self):
        assert self._buffer_pool is not None
        self._prefetch_in_capture = torch.npu.is_current_stream_capturing()
        fork_event = torch.npu.Event()
        torch.npu.current_stream().record_event(fork_event)
        self.copy_stream.wait_event(fork_event)
        with torch.npu.stream(self.copy_stream):
            for name, offloader in self._param_offloaders.items():
                cpu_storage = offloader._cpu_storage
                gpu_buffer = offloader._gpu_buffer
                assert cpu_storage is not None
                assert gpu_buffer is not None
                gpu_buffer.copy_(cpu_storage, non_blocking=True)
        self._copy_done_event.record(self.copy_stream)
        self._event_valid_for_eager = not torch.npu.is_current_stream_capturing()
