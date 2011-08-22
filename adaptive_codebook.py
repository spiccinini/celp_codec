import numpy as np

class AdaptiveCodebook(object):
    def __init__(self, vector_size, cb_size, min_period, np=True):
        self.vector_size = vector_size
        self.cb_size = cb_size
        self.min_period = min_period
        self.max_period = min_period + cb_size

        self.np = np
        self.samples = [0] * self.max_period

    def add_vector(self, vector):
        # Drop last vector_size samples and add vector to top
        self.samples.extend(vector)
        self.samples = self.samples[len(vector):]

    def __len__(self):
        return self.cb_size

    def __getitem__(self, key):
        if key >= self.cb_size:
            raise IndexError

        pos = key + self.min_period
        if pos > self.vector_size:
            ret = self.samples[-pos:-pos+self.vector_size]
        else:
            last = self.samples[-pos:]
            ret = last + last[:self.vector_size-pos]
        if self.np:
            ret = np.asarray(ret,"f")
        return ret

