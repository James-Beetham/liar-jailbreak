
import pathlib,typing,dataclasses,json,logging,random

log = logging.getLogger('utils.dump')

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
  q_plex:typing.Optional[float]=None
  q_time:typing.Optional[float]=None
  y_time:typing.Optional[float]=None
  success:typing.Optional[bool]=None
  @property
  def xpq(self): return repr(eval(self.x)+eval(self.q))
class Dump:
  def __init__(self,dump:list[DumpItem]=[],pth:typing.Optional[pathlib.Path]=None,
               rand:random.Random|None=None):
    self.rand = rand
    if pth is not None:
      with open(pth,'r',encoding='utf-8') as f:
        try: data = json.load(f)
        except: 
          f.seek(0)
          data = [json.loads(v) for v in f.read().split('\n') if len(v)>0]
      if len(data) == 0: log.warning(f'Loaded 0 items from dump file')
      else: log.info(f'Loaded {len(data)} items from dump file.')
      if len(dump)>0: log.warning(f'Created dump with list and path, overwriting list.')
      dump = [DumpItem(**v) for v in data]

    if rand is not None: dump = rand.sample(dump,len(dump))
    self.list = dump

  def get(self,**kwargs):
    """Get a new Dump with only items matching the kwargs provided.

    E.g. idx=1 will select only items where idx is 1. \\
    E.g. idx=[1,2,3] will select items where idx is 1, 2, or 3.
    """
    ret:list[DumpItem] = []
    for di in self.list:
      all_true = True
      for k,v in kwargs.items():
        if type(v) is list:
          all_true = 0<sum([w==di.__dict__[k] for w in v])
        else: all_true = v==di.__dict__[k]
        if not all_true: break
      if not all_true: continue
      ret.append(di)
    return Dump(ret)
  def values(self,key:str): return set(v.__dict__[key] for v in self.list)
  def __getitem__(self,idx): return self.get(idx=idx)
  def __len__(self): return len(self.list)
  def json(self): return [dataclasses.asdict(v) for v in self.list]
  def save(self,pth:pathlib.Path):
    with open(pth,'a',encoding='utf-8') as f: json.dump(self.json(),f)
