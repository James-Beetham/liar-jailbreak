import argparse,os,json,ast


def main(args):
  ctx = Context(args)
  data = ctx.collect()
  print(f'collected: {len(data)} idxes.')

class Context():
  def __init__(self,args):
    self.start_path:str = args.start_path
    self.show_full_path = args.show_full_path

  class ResultPaths:
    def __init__(self,start_path,idx=0):
      self.p = start_path
      self.idx = idx
    @property
    def successes(self): return os.path.join(f'{self.p}{self.idx}','data/success.txt')
    @property
    def fails(self): return os.path.join(f'{self.p}{self.idx}','data/fails.txt')
    @property
    def paths(self): return [self.successes,self.fails]
    @property
    def invalid_paths(self):
      return [v for v in self.paths if not os.path.isfile(v)]
    @property
    def valid(self):
      return len(self.invalid_paths) == 0
    def increment(self):
      self.idx += 1
      return self.valid
    def update(self,idx):
      self.idx = idx
      return self.valid
    def read(self,filepath):
      with open(filepath,'r') as f: data = f.readlines()
      ret = []
      for line_idx,raw in enumerate(data):
        if len(raw)==0: continue
        try: ele = json.loads(raw)
        except: 
          print(f'Misformed json for line: {line_idx}\n{filepath}')
          continue
        if type(ele) is not dict:
          print(f'Misformed json (not dict) for line: {line_idx}\n{filepath}')
          continue
        ret.append({k:v if type(v) is not str else ast.literal_eval(v) for k,v in ele.items()})
      return ret

  def collect(self):
    rp = Context.ResultPaths(self.start_path,idx=-1)
    data = []
    while rp.increment():
      fails = rp.read(rp.fails)
      successes = rp.read(rp.successes)
      if len(fails)+len(successes) == 0:
        print(f'No successes or fails for idx: {rp.idx}') 
        continue
      shared = fails[0] if len(fails)>0 else successes[0]
      data.append(dict(
        idx=rp.idx,
        x=shared['x'],
        yt=shared['yt'],
        qs=[v['q'] for v in successes],
        qf=[v['q'] for v in fails],
        si=[v['i'] for v in successes],
        fi=[v['i'] for v in fails],
        ys=[v['y'] for v in successes],
        yf=[v['y'] for v in fails],
      ))
    failed_paths = rp.invalid_paths
    if self.show_full_path: failed_paths = [os.path.abspath(v) for v in failed_paths]
    print(f'Stopped at idx={rp.idx}. Invalid paths:\n'+'\n'.join(failed_paths))
    return data

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--start_path',type=str,required=True)
  parser.add_argument('--show_full_path',type=bool,default=False,action='store_true')
  args = parser.parse_args()
  main(args)
