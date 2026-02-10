import dataclasses,typing

@dataclasses.dataclass
class BaseDefence:
  name: str
  enable: bool
  mode: str
  n: int
  length: int
  sigma: float