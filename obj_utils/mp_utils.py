import torch.multiprocessing as mp

class Barrier:
    def __init__(self, num_workers):
        self.lock = mp.Lock()
        self.flag = mp.Value("b", False)

    def get(self):
        with self.lock:
            return self.flag.value

    def switch(self):
        with self.lock:
            self.flag.value = (not self.flag.value)