import torch


def dt_dqn(s, a, r, ns, d, q_local, q_target, gamma):
    """Calculate temporal-difference delta_t using fixed Q-targets. This works for batches of tuples.

    Parameters
    ----------
    s : torch.tensor
        Current state (size [n * m], where n is the batch size and m is the number of features in the state)
    a : torch.tensor
        Action taken (size [n])
    r : torch.tensor
        Reward obtained (size [n])
    ns : torch.tensor
        Next state (size [n * m])
    d : torch.tensor
        True if the episode ended after the action (size [n])
    q_local : torch.nn.Module
        Network used to determine the policy
    q_target : torch.nn.Module
        Copy of q_local that is updated less frequently
    gamma : float
        Weight of the estimation of future rewards, in range [0, 1]

    Returns
    -------
        torch.tensor of size [n] where each row is the delta_t of a tuple
    """
    with torch.no_grad():
        QT = q_target(ns).max(1)[0]
    QL = q_local(s).gather(1, a.unsqueeze(1)).squeeze(1)
    return r + gamma * QT * (1 - d) - QL


def dt_double_dqn(s, a, r, ns, d, q_local, q_target, gamma):
    """Calculate temporal-difference delta_t using Double DQN. This works for batches of tuples.

    Parameters
    ----------
    s : torch.tensor
        Current state (size [n * m], where n is the batch size and m is the number of features in the state)
    a : torch.tensor
        Action taken (size [n])
    r : torch.tensor
        Reward obtained (size [n])
    ns : torch.tensor
        Next state (size [n * m])
    d : torch.tensor
        True if the episode ended after the action (size [n])
    q_local : torch.nn.Module
        Network used to determine the policy
    q_target : torch.nn.Module
        Copy of q_local that is updated less frequently
    gamma : float
        Weight of the estimation of future rewards, in range [0, 1]

    Returns
    -------
        torch.tensor of size [n] where each row is the delta_t of a tuple
    """
    with torch.no_grad():
        QLns = q_local(ns).max(1)[1].unsqueeze(1)
        QT = q_target(ns).gather(1, QLns).squeeze(1)
    QL = q_local(s).gather(1, a.unsqueeze(1)).squeeze(1)
    return r + gamma * QT * (1 - d) - QL
