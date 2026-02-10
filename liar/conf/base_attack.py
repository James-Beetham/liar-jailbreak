import dataclasses

import liar.conf.method.base
import liar.conf.model.base_llm

@dataclasses.dataclass
class Data:
  data_dir:str
  test_prefixes_pth:str
  affirmative_prefixes_pth:str
  just_test:bool
  jailbreakbench:bool
  donotanswer:bool
  maliciousinstruct:bool

@dataclasses.dataclass
class Log:
  disable_tqdm:bool
  tqdm_iter_time:float

@dataclasses.dataclass
class BaseAttack:
  adv_llm: liar.conf.model.base_llm.BaseLLM
  target: liar.conf.model.base_llm.BaseLLM
  method: liar.conf.method.base.Method
  repeat: int
  debug:bool
  verbose:bool
  log:Log
  seed:int
  output_dir:str
  data: Data
