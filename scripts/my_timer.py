import time
from contextlib import contextmanager


@contextmanager
def timer(task_name):
    start = time.time()
    yield
    end = time.time()
    print(f"{task_name} completed in {end - start:.4f} seconds")
