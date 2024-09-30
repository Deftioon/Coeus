import numpy as np

class autograd:
    def __init__(self, data, parents=None, op=None):
        self.data = np.array(data)
        self.grad = np.zeros(self.data.shape)
        self.parents = parents or []
        self.op = op

    def __add__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data + other.data,
            parents=[(self, lambda _: 1), (other, lambda _: 1)], op='+', back_func=lambda v: v
        )