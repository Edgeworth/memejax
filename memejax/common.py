from collections.abc import Callable
from dataclasses import fields
from typing import Any, TypeVar

import chex
import jax
import portpicker
import math
from jax import Array, config

ArrayMap = dict[str, Array]
ArrayOrMap = TypeVar("ArrayOrMap", Array, ArrayMap)
ModelOutput = Any

_ENABLE_CHEXIFY = False


def init_jax() -> None:
    # For checkpointing.
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(f"localhost:{port}", num_processes=1, process_id=0)
    # See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions
    config.update("jax2tf_default_native_serialization", False)
    jax.random.PRNGKey(0)  # work around https://github.com/google/jax/issues/16107


def enable_chexify() -> None:
    global _ENABLE_CHEXIFY  # noqa: PLW0603
    _ENABLE_CHEXIFY = True


def use_chexify() -> bool:
    return _ENABLE_CHEXIFY


def jax_enable_debug(enable: bool = True, check_leaks: bool = False) -> None:
    """Enable or disable JAX debugging."""
    config.update("jax_debug_infs", enable)
    config.update("jax_debug_nans", enable)
    config.update("jax_enable_checks", enable)
    config.update("jax_log_checkpoint_residuals", enable)

    check_leaks = enable and check_leaks
    config.update("jax_check_tracer_leaks", check_leaks)
    config.update("jax_include_full_tracebacks_in_locations", enable)

    enable_chexify()


def apply_arrayormap(
    x: ArrayOrMap, train: bool, f: Callable[[str, Array, bool], Array]
) -> ArrayOrMap:
    if isinstance(x, dict):
        return {k: f(k, v, train) for k, v in x.items()}
    return f("array", x, train)


def maybe_chexify(fn: Callable[..., Any]) -> Callable[..., Any]:
    if use_chexify():
        return chex.chexify(fn)
    return fn


def dataclass_has_field(cls: type, field_name: str) -> bool:
    return any(f.name == field_name for f in fields(cls))

def ceil_log(v: float, base: int) -> int:
    return int(math.ceil(math.log(v, base)))


def floor_log(v: float, base: int) -> int:
    return int(math.floor(math.log(v, base)))
