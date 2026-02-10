import os,typing


def load_env(data_dirs:typing.Optional[dict[str,str]]=None)->dict[str,str]:
  """Load the env variables from the .env file.

  Args:
      data_dirs (typing.Optional[dict[str,str]], optional): 
          Dict of env variable name as key, and default paths
          starting in DATA_DIR as values. If keys set in .env file overwrites these
          default paths. Defaults to None.

  Example:
  ```
  >>> ENV = load_env(dict(DATA_DIR_SEQGEO='path/to_seqgeo'))
  >>> ENV['DATA_DIR_SEQGEO']
  'path_to_DATA_DIR/path/to_seqgeo'
  ```

  Raises:
      ValueError: Missing variable from .env file or invalid variable content.

  Returns:
      dict[str,typing.Optional[str]]: Dict of environment variables.
  """

  """

  Args:
      data_dirs (dict[str,str]): 

  Example:
  ```
  >>> ENV = load_env(dict(DATA_DIR_SEQGEO='path/to_seqgeo'))
  >>> ENV['DATA_DIR_SEQGEO']
  'path_to_DATA_DIR/path/to_seqgeo'
  ```

  Returns:
      _type_: Dict of environment variables.
  """

  import dotenv
  env_config = dotenv.dotenv_values()
  env_dict = {k:v for k,v in env_config.items() if v is not None}
  if 'DATA_DIR' not in env_dict: raise ValueError(f'Missing environment variable DATA_DIR in .env.')
  DATA_DIR = env_dict['DATA_DIR']
  if not os.path.isdir(DATA_DIR): raise ValueError(f'[.env] Invalid DATA_DIR:\n\t{DATA_DIR}')


  if data_dirs is not None:
    for k,subpath in data_dirs.items():
      if k not in env_dict: env_dict[k] = os.path.join(DATA_DIR,subpath)

  return env_dict
