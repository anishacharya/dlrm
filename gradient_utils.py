import numpy as np
import torch


def flatten_grads(learner) -> np.ndarray:
    """ Given a model flatten all grads and return as np array """
    flat_grad = []
    for w in learner.parameters():
        grad = w.grad.data
        if grad.is_sparse:
            grad = grad.to_dense()
        flattened = [torch.reshape(grad, (-1,)).numpy()]
        flat_grad.append(flattened)
    return np.array(flat_grad)
