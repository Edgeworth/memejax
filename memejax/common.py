import math
from dataclasses import fields


def dataclass_has_field(cls: type, field_name: str) -> bool:
    return any(f.name == field_name for f in fields(cls))


def ceil_log(v: float, base: int) -> int:
    return int(math.ceil(math.log(v, base)))


def floor_log(v: float, base: int) -> int:
    return int(math.floor(math.log(v, base)))
