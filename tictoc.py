import time
import functools

def tictoc(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs):
        t1 = time.time()
        result = func(*args, **kargs)
        t2 = time.time() - t1
        print(f'{func.__name__} ran in {round(t2, ndigits=2)} seconds')
        return result
    return wrapper