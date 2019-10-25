import math
import numpy as np
import torch

from source.mfa.mfa import MFA


def init_raw_parms_np(K, d, l):
    N_PI = np.zeros([K], dtype=np.float32)  # The mixing coefficient logits
    N_MU = np.zeros([K, d], dtype=np.float32)  # Mean values
    N_A = np.zeros([K, d, l], dtype=np.float32)  # Scale - rectangular matrix
    N_D = np.zeros([K, d], dtype=np.float32)  # Isotropic noise
    return N_PI, N_MU, N_A, N_D


def init_raw_parms_from_gmm(gmm, device="cuda"):
    K = len(gmm.components)
    d, l = gmm.components[0]['A'].shape
    N_PI, N_MU, N_A, N_D = init_raw_parms_np(K, d, l)

    for i, c in gmm.components.items():
        N_PI[i] = np.log(c['pi'])
        N_MU[i, ...] = c['mu']
        N_A[i, ...] = c['A']
        if 's' in c.keys():
            N_D[i, ...] = np.sqrt(c['s'])
        else:
            N_D[i, ...] = np.sqrt(c['D'])

    return torch.tensor(N_PI, requires_grad=True, device=device), \
           torch.tensor(N_MU, requires_grad=True, device=device), \
           torch.tensor(N_A, requires_grad=True, device=device), \
           torch.tensor(N_D, requires_grad=True, device=device)


def raw_to_gmm(PI, MU, A, D, raw_as_log=False):
    K = np.size(PI)
    pi_vals = np.exp(PI) / np.sum(np.exp(PI))
    components = {}
    for i in range(K):
        if raw_as_log:
            components[i] = {'pi': pi_vals[i], 'mu': MU[i, ...], 'A': A[i, ...], 'D': np.exp(-1.0 * D[i])}
        else:
            components[i] = {'pi': pi_vals[i], 'mu': MU[i, ...], 'A': A[i, ...], 'D': np.power(D[i], 2.0)}
    return MFA(components)


def get_per_components_log_likelihood(X, PI_logits, MU, A, sqrt_D):
    """
    Calculate the data log likelihood for low-rank Gaussian Mixture Model.
    See "Learning a Low-rank GMM" (Eitan Richardson) for details
    """
    K, d, l = A.shape

    # Shapes: A[K, d, l], AT[K, l, d], iD[K, d, 1], L[K, l, l]
    AT = A.permute(0, 2, 1)
    iD = sqrt_D.clamp(1e-3, 1.0).pow(-2.0).view(K, d, 1)
    L = torch.eye(l).repeat(K, 1, 1).to(X.device) + torch.bmm(AT, iD * A)
    iL = torch.inverse(L)

    # Calculate Mahalanobis distance
    k_X_c = (X.view(1, -1, d) - MU.view(K, 1, d)).permute(0, 2, 1)  # K x d x m
    k_m_d = (iD * k_X_c) - torch.bmm((torch.bmm((iD * A), iL)), (torch.bmm(AT, (iD * k_X_c))))  # K x d x m
    m_d = (k_X_c * k_m_d).sum(dim=1)  # K x m

    # Shapes: m_d[K, m], log_det_Sigma[K], component_log_probs[K, 1], log_prob_data_given_components[K, m]
    det_L = L.logdet()
    log_det_Sigma = det_L - torch.log(iD.view(K, d)).sum(dim=1)
    log_prob_data_given_components = -0.5 * (
                (d * torch.log(torch.tensor(2.0 * math.pi)) + log_det_Sigma).view(K, 1) + m_d)
    component_log_probs = torch.log(PI_logits.softmax(dim=0)).view(K, 1)
    return component_log_probs + log_prob_data_given_components


def get_log_likelihood(X, PI, MU, A, D):
    comp_LLs = get_per_components_log_likelihood(X, PI, MU, A, D)
    LLs = torch.logsumexp(comp_LLs, dim=0)
    return LLs.sum()


def get_max_posterior_component(X, PI, MU, A, D):
    comp_LLs = get_per_components_log_likelihood(X, PI, MU, A, D)
    return comp_LLs.argmax(dim=0)


def get_latent_posterior_mean(X, c_i, MU, A, sqrt_D):
    """
    Calculate the posterior probability of the latent variable z given x, for selected Gaussians
    """
    K, d, l = A.shape

    # Shapes: A[K, d, l], AT[K, l, d], iD[K, d, 1], L[K, l, l]
    AT = A.permute(0, 2, 1)
    iD = sqrt_D.clamp(1e-3, 1.0).pow(-2.0).view(K, d, 1)
    L = torch.eye(l).repeat(K, 1, 1).to(X.device) + torch.bmm(AT, iD * A)
    iL = torch.inverse(L)

    # Shapes: X_c[m, d, 1], iD_c[m, d, 1]
    X_c = (X - MU.index_select(0, c_i)).view(-1, d, 1)
    iD_c = iD.index_select(0, c_i).view(-1, d, 1)
    m_d_1 = (iD_c * X_c) - ((iD_c * A.index_select(0, c_i)) @ iL.index_select(0, c_i)) @ (AT.index_select(0, c_i) @ (iD_c * X_c))
    mu_z = AT.index_select(0, c_i) @ m_d_1
    return mu_z


def generate_from_posterior(X, PI_logits, MU, A, sqrt_D):
    K, d, l = A.shape

    # Find the most probable components for each sample
    c_i = get_max_posterior_component(X, PI_logits, MU, A, sqrt_D)

    # Calculate the posterior mean z values and use them to generate samples from the model
    z_mu_i = get_latent_posterior_mean(X, c_i, MU, A, sqrt_D)

    # Generate new samples from the posterior mean (ignoring the posterior covariance)
    sample = A.index_select(0, c_i) @ z_mu_i + MU.index_select(0, c_i).view(-1, d, 1)
    return sample.view(sample.shape[0], -1)