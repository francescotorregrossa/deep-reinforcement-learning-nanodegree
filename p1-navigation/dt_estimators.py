import torch


def dt_dqn(s, a, r, ns, d, q_local, q_target, gamma):
    with torch.no_grad():
        QT = q_target(ns).max(1)[0]
    QL = q_local(s).gather(1, a.unsqueeze(1)).squeeze(1)
    return r + gamma * QT * (1 - torch.FloatTensor(d)) - QL


def dt_double_dqn(s, a, r, ns, d, q_local, q_target, gamma):
    pass


"""
QL [5 x 4]
            a_0      a_1       a_2     a_3
tensor([[ 0.0060, -0.0931,  0.0496, -0.0626],    # 0 in the batch
        [ 0.0139, -0.0952,  0.0470, -0.0571],    # 1 
        [ 0.0024, -0.0916,  0.0288, -0.0674],    # 2
        [-0.0207, -0.1126,  0.0453, -0.1079],    # 3
        [ 0.0089, -0.1415,  0.0422, -0.1021]])   # 4

a [5] 
tensor([1, 3, 1, 0, 2])

a.unsqueeze(1) [5 x 1]
tensor([[ 1],
        [ 3],
        [ 1],
        [ 0],
        [ 2]])

QL.gather(1, a.unsqueeze(1)) [5 x 1]
tensor(1.00000e-02 *
        [[-9.3088],
        [-5.7069],
        [-9.1603],
        [-2.0720],
        [ 4.2235]])

QL.squeeze [5]
tensor([-0.0931, -0.0571, -0.0916, -0.0207, 0.0422])
"""
