import numpy as np

class wrapper(): # easy way to store numpy ndarrays and dicts in pandas
    a = np.empty(1)
    def __init__(self, a):
        self.a = a
    def __str__(self):
        try:
            return 'a' + 'x'.join([str(a) for a in self.a.shape])
        except:
            return str(self.a)

