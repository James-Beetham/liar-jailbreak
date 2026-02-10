import csv,os,pathlib,typing

from torch.utils.data import DataLoader, Dataset
import numpy as np

from liar.utils.env import load_env
ENV = load_env(dict(
  DATA_DIR_ADVBENCH='advbench'
))

# From https://github.com/facebookresearch/advprompter/blob/main/utils.py
def load_csv(pth):
  with open(pth) as f:
    dict_reader = csv.DictReader(f)
    csv_list = list(dict_reader)
  return csv_list

class AdvPromptDataset(Dataset):
  def __init__(self, data_pth, augment=False):
    self.dataset = load_csv(data_pth)
    self.augment = augment
    self.augmentations = [
      (0.5,'target',lambda s: s.replace("Sure, here is", "Sure, here's")),
      (0.5,'target',lambda s: s.replace("Sure, h", "H")),
    ]

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx)->dict[str,str]: # instruct,target are the keys
    ret = self.dataset[idx]
    ret = {**ret}
    if self.augment:
      for p,k,f in self.augmentations:
        if np.random.random() < p: ret[k] = f(ret[k])

    return self.dataset[idx]

class AugmentDataLoader(DataLoader):
  def __init__(self, dataset:Dataset, batch_size:int, augment_target:bool, shuffle:bool):
    super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    self.effective_dataset_size = len(self.dataset) # type: ignore
    self.aufgment_target = augment_target

    self.process_fn = lambda s: s.replace("Sure, here is", "Sure, here's")
    self.process_fn2 = lambda s: s.replace("Sure, h", "H")

  def __iter__(self): # type: ignore
    for batch in super(AugmentDataLoader, self).__iter__():
      if self.aufgment_target:
        targets = []
        for target in batch["target"]:
          if np.random.random() < 0.5:
            target = self.process_fn(target)
          if np.random.random() < 0.5:
            target = self.process_fn2(target)
          targets.append(target)
        batch["target"] = targets
      yield batch


SPLITS = dict(
  test='test.csv',
  train='train.csv',
  val='validation.csv',
  jailbreakbench='jailbreakbench.csv',
  donotanswer='donotanswer.csv',
  maliciousinstruct='maliciousinstruct.csv',
  none='none.csv',
)
def get_dataset(split:str,data_dir=None):
  if split not in SPLITS: raise ValueError(f'Unknown advbench split: {split}')
  if data_dir is None:  data_dir = ENV['DATA_DIR_ADVBENCH']
  if type(data_dir) is str: data_dir = pathlib.Path(data_dir)
  data_dir = typing.cast(pathlib.Path,data_dir)
  data_path = data_dir/f'harmful_behaviors/dataset/{SPLITS[split]}'
  dataset = AdvPromptDataset(data_pth=data_path,augment=split=='train')
  return dataset

def get_censored_list(data_dir:typing.Optional[str|pathlib.Path]=None,expanded=True):
  if data_dir is None: data_dir = ENV['DATA_DIR_ADVBENCH']
  if type(data_dir) is str: data_dir = pathlib.Path(data_dir)
  data_dir = typing.cast(pathlib.Path,data_dir)
  prefix_path = data_dir/'test_prefixes.csv'
  prefixes = load_csv(prefix_path)
  prefixes = [list(prefixes[0].keys())[0],*[list(v.values())[0] for v in prefixes]]
  if expanded:
    for pre in list(prefixes):
      if "'" in pre: prefixes.append('’'.join(pre.split("'")))
  return prefixes


def get_dataloader(split, batch_size=2, shuffle=False, augment_target=False):
  dataset = get_dataset(split)
  dataloader = AugmentDataLoader(
    dataset, augment_target=augment_target, shuffle=shuffle, batch_size=batch_size
  )
  return dataloader

