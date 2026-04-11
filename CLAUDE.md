# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

vLLM Ascend (`vllm-ascend`) is a community maintained hardware plugin for running vLLM seamlessly on the Ascend NPU. It implements a hardware-pluggable interface that decouples Ascend NPU integration from upstream vLLM.

**Key Prerequisites:**
- Hardware: Atlas 800I A2/A3, Atlas A2/A3 Training series, Atlas 300I Duo (Experimental)
- OS: Linux
- Python >= 3.10, < 3.12
- CANN == 8.5.1
- PyTorch == 2.9.0, torch-npu == 2.9.0
- Matching vLLM version

## Common Development Commands

### Installation & Build

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Build with custom SOC version (if npu-smi not available)
export SOC_VERSION="ascend910b1"  # A2
export SOC_VERSION="ascend910_9391"  # A3
export SOC_VERSION="ascend310p1"  # 310P
pip install -e .

# Build with verbose output
export VERBOSE=1
pip install -e .

# Skip custom kernel compilation (for CPU-only environments)
export COMPILE_CUSTOM_KERNELS=0
pip install -e .
```

### Code Formatting & Linting

```bash
# Run all formatters and linters
bash format.sh

# Run in CI mode (for pre-commit checks)
bash format.sh ci

# Run ruff check
ruff check vllm_ascend/

# Run ruff format
ruff format vllm_ascend/
```

### Testing

```bash
# Run specific unit test
pytest -sv tests/ut/ops/test_prepare_finalize.py

# Run specific unit test by name
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# Run NPU-specific tests (requires NPU hardware)
pytest -sv tests/e2e/singlecard/test_piecewise_res_consistency

# Run all tests
pytest tests/
```

## Code Architecture

### Plugin Architecture

vLLM Ascend is a hardware plugin that integrates via vLLM's pluggable interface. The entry points are defined in `setup.py`:

- `vllm.platform_plugins`: Registers `NPUPlatform`
- `vllm.general_plugins`: Registers KV connector, model loader, and service profiling

### Directory Structure

```
vllm_ascend/
├── __init__.py              # Plugin registration
├── envs.py                  # Centralized environment variables
├── ascend_config.py         # Ascend configuration
├── platform.py              # NPUPlatform implementation
├── worker/
│   ├── model_runner_v1.py   # v1 model runner
│   └── v2/model_runner.py   # v2 model runner
├── _310p/                    # 310P-specific implementations
│   └── model_runner_310p.py
├── patch/                    # Patches to upstream vLLM
│   ├── platform/            # Platform-level patches
│   └── worker/              # Worker/model-level patches
├── attention/               # NPU attention implementations
├── compilation/             # ACL graph compilation
├── distributed/             # Distributed communication (HCCL, KV transfer)
├── quantization/            # Quantization methods (W8A8, etc.)
└── ops/                     # Custom NPU operators
```

### Key Architectural Patterns

**1. Environment Variables**
- All environment variables are defined in `vllm_ascend/envs.py`
- Use `from vllm_ascend import envs` to access them
- Never hardcode environment variable names in code

**2. Patching**
- Model-specific changes go in `vllm_ascend/patch/`
- Platform patches: `patch/platform/`
- Worker/model patches: `patch/worker/`
- All patches require architectural review

**3. Model Runners**
- `vllm_ascend/worker/model_runner_v1.py` - v1 API
- `vllm_ascend/worker/v2/model_runner.py` - v2 API
- `vllm_ascend/_310p/model_runner_310p.py` - 310P specific
- Changes to model runners require strict review

### Critical NPU Considerations

**1. Avoid `tensor.item()` in Hot Paths**
- `tensor.item()` causes CPU-NPU synchronization
- Use device-side operations when possible
- Batch operations if `item()` is unavoidable
- Requires review if used in performance-critical paths

**2. Memory and Performance**
- Minimize CPU-NPU memory transfers
- Prefer in-place operations where safe
- Monitor memory fragmentation
- Test on actual NPU hardware (Ascend 910B/C)

## Development Guidelines

### Branch Strategy

- **main**: Main branch, tracks vLLM main, CI verified
- **releases/vX.Y.Z**: Release branches for specific vLLM versions
- **rfc/feature-name**: Feature branches for collaboration

### Commit Messages

Follow Conventional Commits with sign-off:

```bash
git commit -s -m "<type>: <summary>" -m "<body>"
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`

### Code Style

- **Imports**: At top, except circular imports/lazy loading
- **No global variables**: Pass dependencies explicitly
- **No magic numbers**: Use named constants
- **Naming**:
  - Classes: `PascalCase`
  - Functions/Variables: `snake_case`
  - Constants: `ALL_UPPER_CASE`

### Adding New Models

Model support is added via patching, not by adding new model files directly. See `vllm_ascend/patch/worker/` for examples.

## Important Environment Variables

Build-time:
- `SOC_VERSION`: Target chip type (ascend910b1, ascend910_9391, ascend310p1)
- `COMPILE_CUSTOM_KERNELS`: Set to 0 to skip kernel compilation
- `CMAKE_BUILD_TYPE`: Release/Debug/RelWithDebugInfo
- `MAX_JOBS`: Number of compile threads
- `ASCEND_HOME_PATH`: CANN toolkit path

Runtime:
- `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`: Enable MatmulAllReduce fusion
- `VLLM_ASCEND_ENABLE_FLASHCOMM1`: Enable FlashComm optimization
- `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`: Enable FLASHCOMM2 with TP group size
- `VLLM_ASCEND_ENABLE_NZ`: Enable FRACTAL_NZ weight format (0=off, 1=quant-only, 2=always)
- `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL`: Enable context parallelism

See `vllm_ascend/envs.py` for complete list with documentation.

## Testing Requirements

- New functionality requires tests in `tests/ut/` or `tests/e2e/`
- Bug fixes require regression tests
- Run all tests locally before submitting PR
- Verify NPU-specific tests on actual hardware

## Related Documentation

- [AGENTS.md](AGENTS.md) - Detailed development guidelines
- [Contributing Guide](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html)
- [Official Documentation](https://docs.vllm.ai/projects/ascend/en/latest/)
- [Versioning Policy](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)
