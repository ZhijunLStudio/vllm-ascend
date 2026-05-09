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
"""Patch create_offloader to return NPU-specific offloaders.

On NPU:
- prefetch backend -> NPUPrefetchOffloader
- uva backend -> NoopOffloader (UVA not supported on NPU)
- auto + offload_group_size > 0 -> NPUPrefetchOffloader
- auto + cpu_offload_gb > 0 -> NoopOffloader (UVA not supported)

Patches at multiple levels to ensure the patched function is used:
1. offloader_base.create_offloader (module definition)
2. offloader_pkg.create_offloader (package re-export)
3. gpu_model_runner.create_offloader (captured via from-import)
"""

import vllm.model_executor.offloader as offloader_pkg
import vllm.model_executor.offloader.base as offloader_base
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import NoopOffloader

from vllm_ascend.offloader.npu_prefetch import NPUPrefetchOffloader

logger = init_logger(__name__)


def _npu_create_offloader(offload_config):
    backend = offload_config.offload_backend
    uva = offload_config.uva
    prefetch = offload_config.prefetch

    if backend == "auto":
        if prefetch.offload_group_size > 0:
            backend = "prefetch"
        elif uva.cpu_offload_gb > 0:
            logger.warning(
                "UVA offloading is not supported on NPU. "
                "Use prefetch backend instead (--offload-backend prefetch "
                "--prefetch.offload-group-size N). Falling back to NoopOffloader."
            )
            return NoopOffloader()
        else:
            return NoopOffloader()

    if backend == "prefetch":
        return NPUPrefetchOffloader(
            group_size=prefetch.offload_group_size,
            num_in_group=prefetch.offload_num_in_group,
            prefetch_step=prefetch.offload_prefetch_step,
            offload_params=prefetch.offload_params,
            mode="cpu",
        )
    elif backend == "uva":
        logger.warning(
            "UVA offloading is not supported on NPU. "
            "Use prefetch backend instead (--offload-backend prefetch "
            "--prefetch.offload-group-size N). Falling back to NoopOffloader."
        )
        return NoopOffloader()
    else:
        return NoopOffloader()


# Patch at all levels to cover from-imports that capture the reference
offloader_base.create_offloader = _npu_create_offloader
offloader_pkg.create_offloader = _npu_create_offloader

# Patch gpu_model_runner module if already imported (captures from-import)
try:
    import vllm.v1.worker.gpu_model_runner as gpu_mr_module

    gpu_mr_module.create_offloader = _npu_create_offloader
except (ImportError, AttributeError):
    pass
