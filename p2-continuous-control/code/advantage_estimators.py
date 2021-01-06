import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def n_step(rewards, dones, future, returns, critic_out, gamma, tau):
    return returns - critic_out


def gae(rewards, dones, future, returns, critic_out, gamma, tau):

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
