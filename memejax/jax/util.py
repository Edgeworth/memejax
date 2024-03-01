from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import chex
import jax
import portpicker
from jax import Array, config

JaxArrayMap = dict[str, Array]
JaxArrayOrMap = TypeVar("JaxArrayOrMap", Array, JaxArrayMap)
JaxModelOutput = Any


_ENABLE_CHEXIFY = False


def init_jax(*, cpu: bool = False) -> None:
    if cpu:
        config.update("jax_platform_name", "cpu")
    # For checkpointing.
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(f"localhost:{port}", num_processes=1, process_id=0)
    # See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions
    config.update("jax2tf_default_native_serialization", True)
    # TODO(1): Bump this when upgrading tensorflow to >=2.16.0
    config.update("jax_serialization_version", 8)
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
    x: JaxArrayOrMap, train: bool, f: Callable[[str, Array, bool], Array]
) -> JaxArrayOrMap:
    if isinstance(x, dict):
        return {k: f(k, v, train) for k, v in x.items()}
    return f("array", x, train)


def maybe_chexify(fn: Callable[..., Any]) -> Callable[..., Any]:
    if use_chexify():
        return chex.chexify(fn)
    return fn


def resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()
