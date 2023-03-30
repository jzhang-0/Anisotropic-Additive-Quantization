import numpy as np
from scipy import integrate


def f(x, d):
    return np.sin(x) ** d

#@profile
def h(x, T):
    if T < np.linalg.norm(x, ord=2):
        d = x.size
        alpha = np.arccos(T / np.linalg.norm(x, ord=2))
        Id, _ = integrate.quad(f, 0, alpha, args=d)
        Id2, _ = integrate.quad(f, 0, alpha, args=d - 2)
        h1 = Id2 - Id
        h2 = Id / (d - 1)
        return h1 * 100, h2 * 100
    else:
        return 0, 0

#@profile
def score_aware_loss(data, weight, Quantization_data, parameter):
    """
    Args:
        data:A column vector
    Returns
        A positive number
    """
    Q_d = Quantization_data
    if parameter.nor:
        s1 = np.vdot(Q_d, data)
        s2 = np.vdot(Q_d, Q_d)
        loss = (parameter.eta - 1) * s1 * s1 + s2 - 2 * parameter.eta * s1 + parameter.eta
        return loss
    else:            
        # I=np.eye(np.size(data))
        h1, h2 = weight

        s1 = np.vdot(Q_d, data)
        # s2=np.linalg.norm(data, ord = 2)**2
        s2 = np.vdot(data, data)
        # s3=np.linalg.norm(Q_d, ord = 2)**2
        s3 = np.vdot(Q_d, Q_d)
        loss = (h1 - h2) * s1 * s1 / s2 + h2 * s3 - 2 * h1 * s1 + h1 * s2
        return loss

def score_aware_loss_one_to_many(data, weight, Quantization_data_batch, parameter):
    """
    Quantization_data_batch:shape = (D,N)
    """
    if parameter.nor:
        # s1 = np.vdot(Q_d, data)
        s1 = data@Quantization_data_batch
        # s2 = np.vdot(Q_d, Q_d)
        s2 = np.sum(Quantization_data_batch**2, axis =0)
        loss = (parameter.eta - 1) * s1 * s1 + s2 - 2 * parameter.eta * s1 + parameter.eta
        return loss

