import numpy as np
import torch
import functools


def flatten_grads(learner) -> np.ndarray:
    """ Given a model flatten all grads and return as np array """
    flat_grad = []
    for w in learner.parameters():
        grad = w.grad.data
        if grad.is_sparse:
            grad = grad.to_dense()
        flattened = torch.reshape(grad, (-1,)).tolist()
        flat_grad.extend(flattened)
    return np.array(flat_grad)


def dist_grads_to_model(grads, learner):
    """ Given Gradients and a model architecture this method updates the model gradients (Corresponding to each param)
    with the supplied grads """
    parameters = learner.to('cpu').parameters()
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = grads[offset:offset + new_size]
        current_data = torch.from_numpy(current_data.reshape(param.shape))
        current_data = current_data.type(param.grad.dtype)
        param.grad = current_data
        offset += new_size
