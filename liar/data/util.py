import pathlib,csv

def load_csv(file_path:pathlib.Path,has_header=True,
             return_cols:list[str]|None=None):
  raise NotImplementedError()
  assert file_path.is_file(),f'Missing csv file: {file_path.absolute()}'
  with open(file_path,encoding='utf-8') as f: 
    if has_header:
      dict_reader = csv.DictReader(f)
    else: raise NotImplementedError()
    csv_list = list(dict_reader)

  if return_cols is not None:
    if len(return_cols) == 1: ret = [v for v in csv_list]
    else: ret = []
  else: ret = csv_list
  return ret

def load_csv_one_col(file_path:pathlib.Path,column_name:str)->list[str]:
  assert file_path.is_file(),f'Missing csv file: {file_path.absolute()}'
  with open(file_path,encoding='utf-8') as f: 
    dict_reader = csv.DictReader(f)
    csv_list = list(dict_reader)
  return [v[column_name] for v in csv_list]

