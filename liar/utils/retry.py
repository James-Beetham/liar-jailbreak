import time

def retry(func,args,kwargs,max_attempts=10,delay=0,raise_error=True,
          fail_print=None,fail_ret=None):
  if max_attempts == 0: return func(*args,**kwargs)
  attempts = max_attempts
  ret = fail_ret
  while attempts>0:
    attempts -= 1
    try:
      ret = func(*args,**kwargs)
      break
    except Exception as e:
      if attempts <= 0 and raise_error: raise e
      else: ret = fail_ret
      if fail_print is not None:
        print(f'Attempt: {max_attempts-attempts+1}')
      if delay>0: time.sleep(delay)
  return ret

