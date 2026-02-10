import dataclasses,typing

import liar.conf.model.defense.base_def

@dataclasses.dataclass
class LoraParams:
  warmstart:bool
  lora_checkpoint:str
  lora_config:dict

@dataclasses.dataclass
class LLMParams:
  device:str
  freeze:bool
  dtype:str
  lora_params:typing.Optional[LoraParams]
  model_name:str
  checkpoint:str
  api: str
  api_key: str

@dataclasses.dataclass
class GenParams:
  max_new_tokens:int
  do_sample:bool
  temperature: float|None
  top_p: float|None



@dataclasses.dataclass
class PromptTemplateItem:
  key:str
  msg:str

@dataclasses.dataclass
class PromptManager:
  xq_wrapper: None|str # str has '{}'
  prompt_template: list[PromptTemplateItem]

@dataclasses.dataclass
class BaseLLM:
  defense: liar.conf.model.defense.base_def.BaseDefence
  batch_size: int
  llm_params: LLMParams
  allow_non_ascii:bool
  gen_params:GenParams
  prompt_manager: PromptManager
