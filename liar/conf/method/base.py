import dataclasses

@dataclasses.dataclass
class Method:
  name:str
  no_q: bool
  mod_prompt: bool
