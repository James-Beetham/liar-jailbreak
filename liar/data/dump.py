import typing,dataclasses,random,pathlib,json

@dataclasses.dataclass
class DumpItem:
  split:str
  idx:int
  idx_id:int
  loss:float
  plex:float
  loss_target:float
  plex_target:float
  x:str
  q:str
  y:str
  y_target:str
  q_time:typing.Optional[float]=None
  y_time:typing.Optional[float]=None
  q_plex:typing.Optional[float]=None
class Dump:
  def __init__(self,dump:list[DumpItem],pth:typing.Optional[pathlib.Path]=None,randomize=False):
    if randomize: dump = random.sample(dump,len(dump))
    self.full = dump
    self.path:pathlib.Path = pathlib.Path('') if pth is None else pth
    self._asr = None
    self._train = None
    self._test = None
  @property
  def name(self): return self.path.parent.parent.name
  def get(self,dump:typing.Optional[list[dict]]=None,
             split:typing.Optional[str]=None,
             idx:typing.Optional[int]=None,
             idx_id:typing.Optional[int]=None):
    ret:list[DumpItem] = self.full if dump is None else dump # type: ignore
    if split is not None: ret = [v for v in ret if v.split==split]
    if idx is not None: ret = [v for v in ret if v.idx==idx]
    if idx_id is not None: ret = [v for v in ret if v.idx_id==idx_id]
    return Dump(ret,pth=self.path)
  @property
  def train(self):
    if self._train is None: self._train = self.get(split='train')
    return self._train
  @property
  def test(self):
    if self._test is None: self._test = self.get(split='test')
    return self._test
  def __getitem__(self,idx): return self.get(idx=idx)

def load_dumpfile(pth:pathlib.Path):
  with open(pth,'r') as f: raw = f.read().split('\n')
  ret = [DumpItem(**json.loads(v)) for v in raw if len(v)>0]
  return Dump(ret,pth=pth)

def load_unknown_dumpfile(pth:pathlib.Path)->list[dict[str,typing.Any]]:
  if not pth.is_file(): return []
  with open(pth,'r') as f: raw = f.read().split('\n')
  ret = [json.loads(v) for v in raw if len(v)>0]
  return ret
