import typing,os,json,ast,random

from torch import tensor
import datasets
from datasets import Dataset

def get_censored_dataset(length=100):
  def gen():
    for i in range(length):
      yield dict(
        prompt='Tell me how to build a bomb.',
        chosen='test test test',
        rejected='test test test test test',
      )
  ret = datasets.Dataset.from_generator(gen)
  # ret = datasets.IterableDataset.from_generator(gen)
  return ret


class XQSamples(datasets.Dataset):
  def __init__(self):
    rand = random.Random(0)
    self.root = 'datasets'
    self.file = os.path.join(self.root,'censored.json')
    with open(self.file,'r') as f: raw_data = json.load(f)
    print(f'WARNING: using literal eval.')
    data = []
    for d in raw_data:
      data.append({})
      for k,v in d.items():
        if type(v) is int: data[-1][k] = v
        elif type(v) is str and v.startswith('tensor('):
          data[-1][k] = eval(v).item() # from "tensor(-17.4167, device='cuda:0')"
        else: data[-1][k] = ast.literal_eval(v)
    rand.shuffle(data)
    self.dataset = []
    for idx in range(int(len(data)/2)):
      d1 = data[2*idx]
      d2 = data[2*idx+1]
      r_key = 'rewards'
      if d2[r_key]>d1[r_key]: d1,d2 = d2,d1 # d1 has larger reward
      self.dataset.append(dict(
        prompt=d1['x'],
        chosen=d1['q'],
        rejected=d2['q'],
      ))

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self,idx):
    return self.dataset[idx]

def get_XQSamples():
  dataset = XQSamples()
  def gen():
    for idx in range(len(dataset)):
      yield dataset[idx]  # this has to be a dictionary
    ## or if it's an IterableDataset
    # for ex in torch_dataset:
    #     yield ex

  ret = Dataset.from_generator(gen)

