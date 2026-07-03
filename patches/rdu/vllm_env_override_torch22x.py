# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
import types

import torch

# torch.Event added in PyTorch 2.3 as a device-agnostic alias for cuda.Event.
# Older versions only have torch.cuda.Event.
if not hasattr(torch, "Event"):
    torch.Event = torch.cuda.Event  # type: ignore[attr-defined]

# # torch.xpu was added in PyTorch 2.0+ for Intel XPU support.
# # On older builds (or non-XPU builds) the attribute is absent, causing
# # AttributeError in xpu_is_initialized(). Provide a minimal stub so the
# # hasattr/is_compiled checks always succeed and return False.
# if not hasattr(torch, "xpu"):
#     class _XpuStub:
#         @staticmethod
#         def _is_compiled() -> bool:
#             return False
#         @staticmethod
#         def is_initialized() -> bool:
#             return False
#         @staticmethod
#         def _is_in_bad_fork() -> bool:
#             return False
#     torch.xpu = _XpuStub()  # type: ignore[attr-defined]

# ── torch._C._cpu._is_amx_tile_supported (added post-2.2) ──────────────────
# Returns False so vLLM skips AMX-tile kernel paths.
if not hasattr(torch._C._cpu, '_is_amx_tile_supported'):
    torch._C._cpu._is_amx_tile_supported = lambda: False  # type: ignore[attr-defined]

# ── torch.compiler.is_compiling (added in 2.3) ─────────────────────────────
# Returns False at eager-mode runtime (correct for CPU inference).
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]

# Some vllm modules access torch.ops._C.<op> and torch.ops.vllm.<op> at module
# level without hasattr guards (e.g. matcher_utils.py). When the vllm._C
# extension is not built these raise AttributeError and prevent importing.
# We register targeted stub attributes ONLY for the specific ops that are
# accessed unconditionally at import time — leaving all the hasattr-guarded
# optional ops (awq_dequantize, gptq_gemm, …) untouched so those guards
# continue to return False and their register_fake calls are correctly skipped.
class _OpPacketStub:
    """Stand-in for an unregistered torch custom op (import-time use only)."""

    def __init__(self, ns: str, name: str) -> None:
        self._ns = ns
        self._name = name

    @property
    def default(self) -> "_OpPacketStub":
        return self

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError(
            f"torch.ops.{self._ns}.{self._name} is not available: "
            "the vllm._C extension is not built for this platform."
        )

    def __repr__(self) -> str:
        return f"<unavailable: torch.ops.{self._ns}.{self._name}>"


def _stub_missing_op(ns_name: str, op_name: str) -> None:
    """Set a stub on torch.ops.<ns_name>.<op_name> only if the op is absent."""
    ns = getattr(torch.ops, ns_name, None)
    if ns is not None and not hasattr(ns, op_name):
        try:
            setattr(ns, op_name, _OpPacketStub(ns_name, op_name))
        except (AttributeError, TypeError):
            pass


# Ops accessed unconditionally at module level in the compilation fusion passes
# (matcher_utils.py, rms_quant_fusion.py, act_quant_fusion.py, …).
for _ns, _op in [
#     # _C ops
#     ("_C", "rms_norm"),
#     ("_C", "fused_add_rms_norm"),
#     ("_C", "rotary_embedding"),
    ("_C", "fused_qk_norm_rope"),
#     ("_C", "silu_and_mul"),
    ("_C", "silu_and_mul_quant"),
#     ("_C", "silu_and_mul_nvfp4_quant"),
#     ("_C", "cutlass_scaled_mm"),
#     ("_C", "static_scaled_fp8_quant"),
#     ("_C", "dynamic_scaled_fp8_quant"),
#     ("_C", "dynamic_per_token_scaled_fp8_quant"),
#     ("_C", "per_token_group_fp8_quant"),
#     ("_C", "scaled_fp4_quant"),
    ("_C", "rms_norm_static_fp8_quant"),
    ("_C", "fused_add_rms_norm_static_fp8_quant"),
    ("_C", "rms_norm_dynamic_per_token_quant"),
    ("_C", "rms_norm_per_block_quant"),
#     # vllm ops
    ("vllm", "flashinfer_rotary_embedding"),
#     ("vllm", "all_gather"),
#     ("vllm", "reduce_scatter"),
    ("vllm", "unified_attention_with_output"),
    ("vllm", "dequant_mxfp4"),
    ("vllm", "quant_dequant_mxfp4"),
    ("vllm", "all_gather"),
    ("vllm", "all_reduce_symmetric_with_copy"),
    ("vllm", "apply_bnb_4bit"),
    ("vllm", "_apply_gguf_embedding"),
    ("vllm", "flashinfer_fp8_blockscale_gemm"),
    ("vllm", "_fused_moe_gguf"),
    ("vllm", "fused_moe_lora"),
    ("vllm", "fused_moe_lora_expand"),
    ("vllm", "fused_moe_lora_shrink"),
    ("vllm", "_fused_mul_mat_gguf"),
    ("vllm", "lora_expand"),
    ("vllm", "lora_shrink"),
    ("vllm", "matmul_mxf4_bf16"),
    ("vllm", "matmul_nvf4_bf16"),
    ("vllm", "moe_forward"),
    ("vllm", "outplace_fused_experts"),
    ("vllm", "patched_fused_scaled_matmul_reduce_scatter"),
    ("vllm", "reduce_scatter"),
    ("vllm", "rocm_per_tensor_float_w8a8_scaled_mm_impl"),
    ("vllm", "rocm_unquantized_gemm"),
    ("vllm", "triton_per_token_group_quant_fp8"),
    ("vllm", "unified_kv_cache_update"),

#     ("vllm", "rocm_unquantized_gemm"),
#     ("vllm", "triton_per_token_group_quant_fp8"),
#     ("vllm", "flashinfer_trtllm_fused_allreduce_norm"),
#     ("vllm", "patched_fused_scaled_matmul_reduce_scatter"),
]:
    _stub_missing_op(_ns, _op)

# torch._C._dynamo is a C extension module (not a Python package), so
# `import torch._C._dynamo.guards` fails even though the attribute exists.
# Pre-register the submodule in sys.modules so dotted imports work.
try:
    import torch._C._dynamo as _dynamo_mod
    for _submod_name in ("guards", "eval_frame", "compiled_autograd"):
        _full_name = f"torch._C._dynamo.{_submod_name}"
        if _full_name not in sys.modules and hasattr(_dynamo_mod, _submod_name):
            sys.modules[_full_name] = getattr(_dynamo_mod, _submod_name)

    # GuardManager was added in newer PyTorch. Provide a stub so that type
    # annotations referencing it (evaluated at function-def time in Python 3.11)
    # don't raise AttributeError. The _compilation_context save/restore logic
    # in wrapper.py will operate on our stub's class attributes at runtime.
    _guards_mod = sys.modules.get("torch._C._dynamo.guards", _dynamo_mod.guards)
    if not hasattr(_guards_mod, "GuardManager"):
        class _GuardManagerStub:
            add_global_state_guard = None
            add_torch_function_mode_stack_guard = None
        _guards_mod.GuardManager = _GuardManagerStub  # type: ignore[attr-defined]
except ImportError:
    pass

# torch.Tag.needs_fixed_stride_order (and potentially other tags) were added in
# newer PyTorch versions. torch._C.Tag is a C extension enum so we cannot add
# attributes directly; instead we shadow torch.Tag with a proxy that delegates
# known attributes to the real enum and returns None for anything missing.
# direct_register_custom_op filters None out of the tags tuple before calling
# lib.define, so unknown tags become no-ops on older PyTorch.
if hasattr(torch, "Tag") and not hasattr(torch.Tag, "needs_fixed_stride_order"):
    _RealTag = torch.Tag

    class _TagProxy:
        def __getattr__(self, name):
            return getattr(_RealTag, name, None)

    torch.Tag = _TagProxy()  # type: ignore[assignment]

# torch.uint16/uint32/uint64 were added in PyTorch 2.3. Third-party packages
# (e.g. lmcache) reference them at module level as dict keys. We cannot create
# real torch.dtype objects without C extensions, so we register hashable
# sentinels. Actual tensor ops using these dtypes will still fail on 2.2.
if not hasattr(torch, "uint16"):
    class _StubDtype:
        """Hashable placeholder for a torch dtype absent in this PyTorch build."""
        def __init__(self, name: str) -> None:
            self._name = name
        def __repr__(self) -> str:
            return f"torch.{self._name}"
        def __hash__(self) -> int:
            return hash(self._name)
        def __eq__(self, other: object) -> bool:
            return isinstance(other, _StubDtype) and self._name == other._name

    for _dtype_name in ("uint16", "uint32", "uint64"):
        if not hasattr(torch, _dtype_name):
            setattr(torch, _dtype_name, _StubDtype(_dtype_name))

# torch.amp.custom_fwd / custom_bwd were added in PyTorch 2.4 and gained a
# `device_type` keyword argument at the same time. Older versions only have
# torch.cuda.amp.custom_fwd/custom_bwd and don't accept device_type.
if hasattr(torch, "amp") and not hasattr(torch.amp, "custom_fwd"):
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
        _cuda_custom_fwd = torch.cuda.amp.custom_fwd
        _cuda_custom_bwd = torch.cuda.amp.custom_bwd

        def _amp_custom_fwd(func=None, *, device_type=None, **kwargs):  # type: ignore[misc]
            if func is not None:
                return _cuda_custom_fwd(func, **kwargs)
            return lambda f: _cuda_custom_fwd(f, **kwargs)

        def _amp_custom_bwd(func=None, **kwargs):  # type: ignore[misc]
            if func is not None:
                return _cuda_custom_bwd(func, **kwargs)
            return lambda f: _cuda_custom_bwd(f, **kwargs)

        torch.amp.custom_fwd = _amp_custom_fwd  # type: ignore[attr-defined]
        torch.amp.custom_bwd = _amp_custom_bwd  # type: ignore[attr-defined]

# torch._inductor.runtime (and its triton_helpers submodule) were added in
# newer PyTorch. Stub them out and wire libdevice from wherever Triton exposes
# it in the installed version (Triton 3.x: triton.language.extra.libdevice).
if "torch._inductor.runtime" not in sys.modules:
    _inductor_runtime = types.ModuleType("torch._inductor.runtime")
    _inductor_runtime.__package__ = "torch._inductor.runtime"
    # __path__ is required for Python to treat this as a package and allow
    # submodule imports like torch._inductor.runtime.triton_heuristics.
    _inductor_runtime.__path__ = []  # type: ignore[attr-defined]
    sys.modules["torch._inductor.runtime"] = _inductor_runtime

    _triton_helpers = types.ModuleType("torch._inductor.runtime.triton_helpers")
    _triton_helpers.__package__ = "torch._inductor.runtime"
    try:
        from triton.language.extra import libdevice as _libdevice
        _triton_helpers.libdevice = _libdevice  # type: ignore[attr-defined]
    except ImportError:
        pass
    sys.modules["torch._inductor.runtime.triton_helpers"] = _triton_helpers
    _inductor_runtime.triton_helpers = _triton_helpers  # type: ignore[attr-defined]

    # CachingAutotuner is used in piecewise_backend.py via isinstance() check.
    _triton_heuristics = types.ModuleType("torch._inductor.runtime.triton_heuristics")
    _triton_heuristics.__package__ = "torch._inductor.runtime"

    class _CachingAutotunerStub:
        """Stub for CachingAutotuner — isinstance checks return False at runtime."""

    _triton_heuristics.CachingAutotuner = _CachingAutotunerStub  # type: ignore[attr-defined]
    sys.modules["torch._inductor.runtime.triton_heuristics"] = _triton_heuristics
    _inductor_runtime.triton_heuristics = _triton_heuristics  # type: ignore[attr-defined]

    try:
        import torch._inductor as _ti
        _ti.runtime = _inductor_runtime  # type: ignore[attr-defined]
    except ImportError:
        pass

# torch._logging._internal.trace_structured was added in newer PyTorch.
# Inject a no-op so the import always succeeds on older versions.
try:
    import torch._logging._internal as _logging_internal
    if not hasattr(_logging_internal, "trace_structured"):
        def _trace_structured_noop(*args, **kwargs):
            pass
        _logging_internal.trace_structured = _trace_structured_noop  # type: ignore[attr-defined]
except ImportError:
    pass

# torch._inductor.custom_graph_pass (and CustomGraphPass within it) were added
# in newer PyTorch. Register a stub module so the import always succeeds.
try:
    import torch._inductor.custom_graph_pass  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _cgp_mod = types.ModuleType("torch._inductor.custom_graph_pass")

    class _CustomGraphPassStub:
        pass

    _cgp_mod.CustomGraphPass = _CustomGraphPassStub  # type: ignore[attr-defined]
    sys.modules["torch._inductor.custom_graph_pass"] = _cgp_mod
    try:
        import torch._inductor as _ti_cgp
        _ti_cgp.custom_graph_pass = _cgp_mod  # type: ignore[attr-defined]
    except ImportError:
        pass

# torch.fx.experimental.symbolic_shapes.statically_known_true was added in
# newer PyTorch; older versions only have definitely_true. Alias it so imports
# always use the canonical name.
try:
    import torch.fx.experimental.symbolic_shapes as _sym_shapes
    if not hasattr(_sym_shapes, "statically_known_true") and hasattr(
        _sym_shapes, "definitely_true"
    ):
        _sym_shapes.statically_known_true = _sym_shapes.definitely_true  # type: ignore[attr-defined]
except ImportError:
    pass

# torch.distributed._symmetric_memory is not present in all PyTorch builds.
# Pre-import it here so that parallel_state.py can assume it's already loaded.
# When the module is absent, register a stub with enable_symm_mem_for_group=None
# so collective_fusion.py can import it without a try/except.
try:
    import torch.distributed._symmetric_memory as _symm_mem
    if not hasattr(_symm_mem, "enable_symm_mem_for_group"):
        _symm_mem.enable_symm_mem_for_group = None  # type: ignore[attr-defined]
except ModuleNotFoundError:
    _symm_mem_stub = types.ModuleType("torch.distributed._symmetric_memory")
    _symm_mem_stub.enable_symm_mem_for_group = None  # type: ignore[attr-defined]
    sys.modules["torch.distributed._symmetric_memory"] = _symm_mem_stub

# torch.distributed.distributed_c10d._unregister_process_group was added in
# newer PyTorch versions. Inject a no-op stub so imports always succeed.
try:
    import torch.distributed.distributed_c10d as _c10d
    if not hasattr(_c10d, "_unregister_process_group"):
        def _unregister_process_group(group_name: str) -> None:
            pass
        _c10d._unregister_process_group = _unregister_process_group  # type: ignore[attr-defined]
except ImportError:
    pass

# torch._higher_order_ops.auto_functionalized is not re-exported from the
# package __init__ in PyTorch 2.2 (it lives in the submodule
# auto_functionalize). Expose it at the package level so callers using
# `from torch._higher_order_ops import auto_functionalized` succeed.
try:
    import torch._higher_order_ops as _hops
    if not hasattr(_hops, "auto_functionalized"):
        from torch._higher_order_ops.auto_functionalize import (
            auto_functionalized as _auto_functionalized,
        )
        _hops.auto_functionalized = _auto_functionalized  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    pass

# torch.library.wrap_triton was added in PyTorch 2.3. It wraps a Triton kernel
# so it can be traced by torch.compile. On older PyTorch (or where Triton is
# unavailable) a passthrough identity is sufficient to allow imports to succeed.
if hasattr(torch, "library") and not hasattr(torch.library, "wrap_triton"):
    torch.library.wrap_triton = lambda fn: fn  # type: ignore[attr-defined]

# PyTorch 2.2's torch/library.py does `tuple(tags)` but callers (e.g.
# torch/distributed/_functional_collectives.py) pass a single torch.Tag
# enum value instead of a sequence, causing:
#   TypeError: 'torch._C.Tag' object is not iterable
# Patch Library.define to normalise tags to a tuple before the C call.
if hasattr(torch, "library") and hasattr(torch.library, "Library"):
    _orig_lib_define = torch.library.Library.define

    def _patched_lib_define(self, schema, alias_analysis="", *, tags=()):
        if isinstance(tags, torch._C.Tag):
            tags = (tags,)
        return _orig_lib_define(self, schema, alias_analysis, tags=tags)

    torch.library.Library.define = _patched_lib_define  # type: ignore[method-assign]

from vllm.logger import init_logger
from vllm.utils.torch_utils import is_torch_equal

logger = init_logger(__name__)

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

# see https://github.com/vllm-project/vllm/issues/10480
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
# see https://github.com/vllm-project/vllm/issues/10619
if hasattr(torch._inductor, "config"):
    torch._inductor.config.compile_threads = 1

# ===================================================
# torch 2.9 Inductor PythonWrapperCodegen monkeypatch
# ===================================================
# This change monkeypatches memory_plan_reuse in pytorch 2.9.0 to work around
# a test failure for test_multi_graph_piecewise_compile_outputs_equal.
# For more context, see https://github.com/pytorch/pytorch/pull/165514.


def memory_plan_reuse_patched(self):
    import torch._inductor.ir as ir
    from torch._inductor.codegen.wrapper import (
        EnterSubgraphLine,
        ExitSubgraphLine,
        MemoryPlanningLine,
        MemoryPlanningState,
        SubgraphPythonWrapperCodegen,
    )
    from torch._inductor.virtualized import V

    def get_output_names(graph_outputs) -> list[str]:
        import itertools

        names = []
        shape_counter = itertools.count(0)
        none_counter = itertools.count(0)
        for node in graph_outputs:
            if isinstance(node, ir.NoneAsConstantBuffer):
                names.append(f"{V.graph.name}_none{next(none_counter)}")
            elif isinstance(node, ir.ShapeAsConstantBuffer):
                names.append(f"{V.graph.name}_shape{next(shape_counter)}")
            else:
                names.append(node.get_name())
        return names

    if (
        isinstance(V.graph.wrapper_code, SubgraphPythonWrapperCodegen)
        and V.graph.wrapper_code.partition_signatures is not None
    ):
        out_names = get_output_names(
            V.graph.wrapper_code.partition_signatures.output_nodes
        )
    else:
        out_names = V.graph.get_output_names()

    while (
        self.lines
        and isinstance(self.lines[-1], MemoryPlanningLine)
        and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
    ):
        # these lines will be pointless
        self.lines.pop()

    # codegen allocations in two passes
    planning_states = [MemoryPlanningState()]
    past_planning_states = []
    for i in range(len(self.lines)):
        line = self.lines[i]
        if isinstance(line, MemoryPlanningLine):
            self.lines[i] = line.plan(planning_states[-1])
        elif isinstance(line, EnterSubgraphLine):
            planning_states.append(MemoryPlanningState())
        elif isinstance(line, ExitSubgraphLine):
            past_planning_states.append(planning_states.pop())
    past_planning_states.append(planning_states.pop())
    assert len(planning_states) == 0


# ===================================================
# torch 2.9 Inductor get_graph_partition_signature monkeypatch
# ===================================================
# This change monkeypatches get_graph_partition_signature in pytorch 2.9.0 to
# fix inductor partition + attention-nvfp4 quant fusion, tested in
# `tests/compile/test_fusion_attn.py::test_attn_quant`.
# For more context, see https://github.com/pytorch/pytorch/pull/165815.


def get_graph_partition_signature_patched(
    self, partitions, skip_cudagraphs: list[bool]
):
    """
    Gets signature for each graph partition, including input nodes, output nodes, and
    whether deallocating an input within graph partition.
    """
    from torch._inductor import dependencies
    from torch._inductor.ir import GraphPartitionSignature, MutationOutput, NoneLayout
    from torch._inductor.virtualized import V
    from torch.utils._ordered_set import OrderedSet

    signatures = []

    unmet_output_names = OrderedSet(V.graph.get_output_names())
    name_to_node = self.get_name_to_nodes()

    def is_none_layout(buf_name: str) -> bool:
        """
        Checks if buf_name is NoneLayout. Buffers with NoneLayout is not allocated
        so graph partition should not take it as inputs or outputs.
        """
        buf = self.name_to_buf.get(buf_name, None)

        if buf is None:
            return False

        if isinstance(buf.node.layout, NoneLayout):
            if isinstance(buf.node, MutationOutput) and (
                real_name := self.mutation_real_name.get(buf_name, None)
            ):
                return is_none_layout(real_name)

            return True

        return False

    for partition, skip_cudagraph in zip(
        reversed(partitions), reversed(skip_cudagraphs)
    ):
        output_names: OrderedSet[str] = OrderedSet()

        for node in partition:
            output_names.update(node.outputs_by_name.keys())

        returned_output_names = output_names.intersection(unmet_output_names)

        # all reads/writes are partition inputs except those generated
        # within the partition and tensor constants
        read_writes = dependencies.ReadWrites.merge_list(
            [node.read_writes for node in partition]
        )

        # WeakDep is fake dependency on unused buffer. It should not appear
        # in partition_input_names for inputs that are actually read or written.
        partition_input_names = (
            OrderedSet(
                [
                    x.name
                    for x in read_writes.reads | read_writes.writes
                    if not is_none_layout(x.name)
                ]
            )
            - output_names
        )

        partition_input_names = OrderedSet(
            self.mutation_real_name.get(name, name) for name in partition_input_names
        )

        buffer_names_to_free: OrderedSet[str] = OrderedSet()
        for node in partition:
            buffer_names_to_free.update(node.last_usage)

        # buffer_names_to_free may contain buffers allocated in previous
        # graph partitions. These buffers should also be a partition
        # input.
        extra_input_names = [
            name
            for name in (buffer_names_to_free - output_names)
            if name in name_to_node
        ]
        partition_input_names.update(extra_input_names)

        input_nodes = {
            name: name_to_node[name]
            for name in partition_input_names
            if name in name_to_node
        }
        input_deallocation = {
            name: name in buffer_names_to_free
            for name in partition_input_names
            if name in name_to_node
        }

        # if an input tensor is not freed in the partition function, it should
        # also be returned as an output. This brings benefits to cudagraph
        # since the returned output tensor is a cudagraph managed tensor with
        # a static tensor address.
        extra_output_names = [
            name
            for name in partition_input_names
            if name in name_to_node and name not in buffer_names_to_free
        ]

        returned_output_names.update(extra_output_names)

        returned_output_names = OrderedSet(
            self.mutation_real_name.get(name, name) for name in returned_output_names
        )

        output_nodes = [
            name_to_node[name]
            for name in returned_output_names
            if not is_none_layout(name)
        ]

        constant_names = [
            name for name in partition_input_names if name in V.graph.constants
        ]

        symbol_inputs = self.get_graph_partition_symbol_inputs(partition, input_nodes)

        partition_signature = GraphPartitionSignature(
            symbol_inputs,
            input_nodes,
            output_nodes,
            input_deallocation,
            skip_cudagraph,
            constant_names,
        )

        signatures.append(partition_signature)

        unmet_output_names = partition_input_names.union(
            unmet_output_names - returned_output_names
        )

    return signatures[::-1]


# ========================================
# torch 2.9 Inductor Scheduler monkeypatch
# ========================================
# This change monkeypatches a function in Inductor to work around the following
# bug: https://github.com/vllm-project/vllm/issues/26678
#
# The bug occurs when `use_inductor_graph_partition` is turned on and there
# exists operators inside of `splitting_ops` that have an in-place mutation. In
# vllm, this specifically occurs on the operator
# vllm.unified_attention_with_output. In this case, inductor does not populate
# the inductor IR's `origin_node` field, causing an assertion error when trying
# to access the node's `origin_node` field.
#
# So, we will monkeypatch torch._inductor.scheduler.Scheduler.should_partition
# so that it does not access the inductor IR node's `origin_node` field and just
# returns True if a node is registered as having a custom partition function.
# This is ok for now since vllm's implementation of the custom partition
# functions just return True.
# ========================================


def should_partition_patched(self, node, should_log: bool = False) -> bool:
    # This is a patched version of
    # torch._inductor.scheduler.Scheduler.should_partition that modifies
    # the following piece of code so that we always return True:
    # https://github.com/pytorch/pytorch/blob/ecb53078faf86ca1b33277df33b82985675bb011/torch/_inductor/scheduler.py#L4712-L4724
    """Return True if we should partition the inductor graph on this node"""

    import torch._inductor.ir as ir
    from torch._inductor.scheduler import (
        BaseSchedulerNode,
        FusedSchedulerNode,
    )
    from torch._inductor.utils import (
        _unstable_customized_partition_wrapper,
        is_cudagraph_unsafe_op,
        maybe_log_cudagraph_partition,
    )

    # Allow users to manually specify if a node should be partitioned
    # Can only do this for FallbackKernels
    ir_node = node.node
    if isinstance(ir_node, torch._inductor.ir.FallbackKernel) and (
        op := ir_node.op_overload
    ):
        op_overload_packet_name = op.name()
        op_overload_name = (
            f"{op_overload_packet_name}.{op._overloadname}"
            if isinstance(op, torch._ops.OpOverload)
            else op_overload_packet_name
        )
        if (
            op_overload_packet_name
            in torch._inductor.config.custom_should_partition_ops
            or op_overload_name in torch._inductor.config.custom_should_partition_ops
        ):
            assert isinstance(op, torch._ops.OpOverload)
            return True

    # When not using cudagraphs, keep all kernels in the `call` function
    # instead of graph partition functions, since graph partition only brings
    # benefit to cudagraph
    if (
        not torch._inductor.config.triton.cudagraphs
        and _unstable_customized_partition_wrapper.wrapper is None
    ):
        return True

    # avoid duplicating logs when should_partition is called multiple times
    # on the same node
    def noop_log(msg: str, node: BaseSchedulerNode | None) -> None:
        return

    log_partition_reason = maybe_log_cudagraph_partition if should_log else noop_log

    if isinstance(node, FusedSchedulerNode):
        return any(self.should_partition(snode) for snode in node.snodes)

    assert node.node is not None

    if not node.is_gpu():
        log_partition_reason("non gpu ops", node=node)

        return True

    if isinstance(node.node, ir.DeviceCopy):
        log_partition_reason("DeviceCopy ops", node=node)
        return True

    if isinstance(node.node, ir.Conditional):
        log_partition_reason("Conditional ops", node=node)
        return True

    if getattr(node.node, "unbacked_bindings", None):
        log_partition_reason("unbacked binding ops", node=node)
        return True

    if is_cudagraph_unsafe_op(node.node):
        log_partition_reason("CUDAGraph-unsafe custom ops", node=node)
        return True

    return False


def _update_scheduler_patched(self) -> None:
    # Copied from torch._inductor.graph.GrahLowering._update_scheduler. Patches
    # this method so that we can patch Scheduler.should_partition with the
    # function above
    """
    (Re)initializes the scheduler member.  When initializing the scheduler, no CUBIN
    files should be generated (to avoid biasing any benchmarks and pessimizing
    fusion decisions).
    """
    import torch._inductor.config as config
    from torch._inductor.scheduler import Scheduler

    Scheduler.should_partition = should_partition_patched
    Scheduler.get_graph_partition_signature = get_graph_partition_signature_patched

    with config.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)


# ===================================================
# torch 2.9 Inductor get_raw_stream workaround
# ===================================================
# Workaround for TorchInductor autotune using get_raw_stream() without defining it.
# This occurs when compile_sizes > 1 in compilation_config.
# For more context, see https://github.com/vllm-project/vllm/issues/30905.
def _patch_get_raw_stream_if_needed():
    """Workaround for TorchInductor autotune get_raw_stream() bug."""
    from vllm.utils.torch_utils import is_torch_equal

    # Only apply the patch for torch 2.9.0 or 2.9.1
    if is_torch_equal("2.9.0") or is_torch_equal("2.9.1"):
        import builtins

        # Check if CUDA functionality is available without initializing CUDA
        # _cuda_getCurrentRawStream only exists in CUDA builds of PyTorch
        if hasattr(torch._C, "_cuda_getCurrentRawStream"):
            from torch._C import _cuda_getCurrentRawStream as _get_raw_stream

            builtins.get_raw_stream = _get_raw_stream  # type: ignore[attr-defined]


_patch_get_raw_stream_if_needed()

if is_torch_equal("2.9.0"):
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.graph import GraphLowering
    from torch.utils._config_module import _Config, _ConfigEntry

    # `custom_should_partition_ops` is a new config after 2.9.0. So this would
    # not overwrite any user configs.
    torch._inductor.config._config["custom_should_partition_ops"] = _ConfigEntry(
        _Config(default=[])
    )

    PythonWrapperCodegen.memory_plan_reuse = memory_plan_reuse_patched
    GraphLowering._update_scheduler = _update_scheduler_patched
