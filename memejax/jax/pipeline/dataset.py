from collections.abc import Iterator
from typing import Protocol

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from jax.typing import ArrayLike

from memejax.jax.util import JaxArrayMap


@struct.dataclass
class JaxTrainData:
    model_inp: JaxArrayMap  # for input data to model - data only necessary to run model
    aux: JaxArrayMap  # for labels, etc - for loss functions etc

    # Declare this here so it isn't recompiled on every call of ordered_batches.
    @staticmethod
    @jax.jit
    def _get_by_idxs(data: JaxArrayMap, idxs: Array) -> JaxArrayMap:
        return {k: v[idxs] for k, v in data.items()}

    def select_by_idxs(self, idxs: Array) -> "JaxTrainData":
        return JaxTrainData(
            model_inp=self._get_by_idxs(self.model_inp, idxs), aux=self._get_by_idxs(self.aux, idxs)
        )


class JaxDataset(Protocol):
    def sample(self, _rng: Array) -> JaxTrainData:
        """Return one single sample."""
        return JaxTrainData({}, {})

    def batch(self, batch_size: int, rng: Array) -> JaxTrainData:
        """Return one batch. Note the default implementation is not very efficient."""
        model_inp: dict[str, list[ArrayLike]] = {}
        aux: dict[str, list[ArrayLike]] = {}

        for _ in range(batch_size):
            rng, subrng = jax.random.split(rng)
            sample = self.sample(subrng)
            for k, v in sample.model_inp.items():
                model_inp.setdefault(k, []).append(v)
            for k, v in sample.aux.items():
                aux.setdefault(k, []).append(v)

        batch_model_inp = {k: jnp.stack(v) for k, v in model_inp.items()}
        batch_aux = {k: jnp.stack(v) for k, v in aux.items()}
        return JaxTrainData(model_inp=batch_model_inp, aux=batch_aux)

    def batches(self, num_batches: int, batch_size: int, rng: Array) -> Iterator[JaxTrainData]:
        """Default implementation. Note the default implementation is not very
        efficient and selects without replacement, reusing samples."""
        assert num_batches >= 0  # don't know the size of the dataset
        for _ in range(num_batches):
            rng, subrng = jax.random.split(rng)
            yield self.batch(batch_size, subrng)

    def num_samples(self) -> int:
        raise NotImplementedError


@struct.dataclass
class JaxArrayMapDataset(JaxDataset):
    """Dataset where each value in the JaxArrayMap has the same first dimension
    which is the number of samples."""

    data: JaxTrainData

    def __post_init__(self) -> None:
        # Check number of examples are consistent.
        num_samples = self.num_samples()
        for v in self.data.model_inp.values():
            assert v.shape[0] == num_samples
        for v in self.data.aux.values():
            assert v.shape[0] == num_samples

    def sample(self, rng: Array) -> JaxTrainData:
        idx = jax.random.randint(rng, shape=(), minval=0, maxval=self.num_samples())
        return self.data.select_by_idxs(idx)

    def batches(self, num_batches: int, batch_size: int, rng: Array) -> Iterator[JaxTrainData]:
        return self.ordered_batches(num_batches, batch_size, rng, random=True)

    def latest_batches(self, num_batches: int, batch_size: int) -> Iterator[JaxTrainData]:
        return self.ordered_batches(
            num_batches, batch_size, jax.random.PRNGKey(0), reverse=True, random=False
        )

    def ordered_batches(
        self,
        num_batches: int,
        batch_size: int,
        rng: Array,
        random: bool = True,
        reverse: bool = False,
    ) -> Iterator[JaxTrainData]:
        num_samples = self.num_samples()
        if num_batches < 0:
            num_batches = num_samples  # large number so it uses all batches
        num_batches = min(num_samples // batch_size, num_batches)
        odd_batch_samples = num_samples % batch_size
        batch_samples = num_batches * batch_size

        if random:
            assert not reverse
            idxs = jax.random.permutation(rng, num_samples)
            odd_idxs = idxs[batch_samples : batch_samples + odd_batch_samples]
            idxs = idxs[:batch_samples]
            idxs = idxs.reshape((num_batches, batch_size))
        elif reverse:
            idxs = jnp.arange(self.num_samples() - 1, -1, -1)
            odd_idxs = idxs[batch_samples : batch_samples + odd_batch_samples]
            idxs = idxs[:batch_samples]
            idxs = idxs.reshape((num_batches, batch_size))
        else:
            idxs = jnp.arange(num_samples)
            odd_idxs = idxs[batch_samples : batch_samples + odd_batch_samples]
            idxs = idxs[:batch_samples]
            idxs = idxs.reshape((num_batches, batch_size))

        # Could do vmap if we let this function return a JaxArrayMap of
        # (num_batches, batch_size, ...), but this is fast enough for now.
        batches = [self.data.select_by_idxs(batch_idxs) for batch_idxs in idxs]
        if odd_batch_samples > 0:
            odd_batch = self.data.select_by_idxs(odd_idxs)
            batches.append(odd_batch)
        return iter(batches)

    def num_samples(self) -> int:
        return len(next(iter(self.data.model_inp.values())))
