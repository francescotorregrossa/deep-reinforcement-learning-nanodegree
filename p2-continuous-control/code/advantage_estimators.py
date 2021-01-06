import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def n_step(rewards, dones, future, returns, critic_out, gamma, tau):
    """Calculate advantages using the n-step method.

    Parameters
    ----------
    rewards : list of torch.tensor
        (Unused) Reward obtained (list of len() n tensors of size [num_agents * 1])
    dones : list of torch.tensor
        (Unused) True if the episode ended after the previous action 
        (list of len() n tensors of size [num_agents * 1])
    future : torch.tensor
        (Unused) tensor of size [num_agents * 1], 
        represents the predicted value of the n-th next state
    returns : torch.tensor
        tensor of size [n * num_agents * 1], 
        represents discounted returns
    critic_out : torch.tensor
        tensor of size [n * num_agents * 1], 
        represents the values predicted by the critic for each state
    gamma : float
        (Unused) Weight of the estimation of future rewards, in range [0, 1]
    tau : float
        (Unused) Used for lambda returns, in range [0, 1]

    Returns
    -------
        torch.tensor of size [n * num_agents * 1]
    """

    return returns - critic_out


def gae(rewards, dones, future, returns, critic_out, gamma, tau):
    """Calculate advantages using the generalized estimation method.

    Parameters
    ----------
    rewards : list of torch.tensor
        Reward obtained (list of len() n tensors of size [num_agents * 1])
    dones : list of torch.tensor
        True if the episode ended after the previous action 
        (list of len() n tensors of size [num_agents * 1])
    future : torch.tensor
        tensor of size [num_agents * 1], 
        represents the predicted value of the n-th next state
    returns : torch.tensor
        (Unused) tensor of size [n * num_agents * 1], 
        represents discounted returns
    critic_out : torch.tensor
        tensor of size [n * num_agents * 1], 
        represents the values predicted by the critic for each state
    gamma : float
        Weight of the estimation of future rewards, in range [0, 1]
    tau : float
        Used for lambda returns, in range [0, 1]

    Returns
    -------
        torch.tensor of size [n * num_agents * 1]
    """

    # create a single tensor for rewards, dones and predicted_values.
    # also, dones and predicted_values have n+1 rows: predictied_values adds G or future at the end
    # whereas dones adds a row of zeros at the beginning; this simplifies access to this data in the loop
    rewards = torch.cat([r.unsqueeze(0) for r in rewards], dim=0)
    dones = torch.cat([torch.zeros([1, 20, 1]).to(device)] +
                      [d.unsqueeze(0) for d in dones], dim=0)
    predictied_values = torch.cat((critic_out, future.unsqueeze(0)), dim=0)

    # retrieve n from a tensor's size, instead of passing it as a parameter
    n = list(critic_out.size())[0]

    # compute A_hats and As backwards, and store the As as you go
    A = torch.zeros([20, 1]).to(device)
    advantages = []
    for i in reversed(range(n)):
        A_hat = rewards[i] + gamma * predictied_values[i+1] * \
            (1 - dones[i+1]) - predictied_values[i] * (1 - dones[i])
        A = A_hat + gamma * tau * A
        advantages.insert(0, A.unsqueeze(0))
    return torch.cat(advantages, dim=0)
