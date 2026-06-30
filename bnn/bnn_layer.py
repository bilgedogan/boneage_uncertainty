import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Parameters for the weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Parameters for the bias distribution
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize mean weights like a standard linear layer
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        
        # Initialize rho such that the initial std is small (e.g., 0.1)
        # sigma = log(1 + exp(rho)) => rho = log(exp(sigma) - 1)
        init_sigma = 0.1
        init_rho = math.log(math.exp(init_sigma) - 1)
        nn.init.constant_(self.weight_rho, init_rho)
        
        # Initialize bias mean
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        
        # Initialize bias rho
        nn.init.constant_(self.bias_rho, init_rho)

    def forward(self, x, sample=True):
        if sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight_epsilon = torch.randn_like(weight_sigma)
            weight = self.weight_mu + weight_sigma * weight_epsilon

            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_epsilon = torch.randn_like(bias_sigma)
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """
        Compute the analytical KL divergence between posterior and prior.
        D_KL( N(mu, sigma^2) || N(0, prior_sigma^2) ) =
            log(prior_sigma / sigma) + (sigma^2 + mu^2) / (2 * prior_sigma^2) - 0.5
        """
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        kl_weight = torch.log(self.prior_sigma / weight_sigma) + \
                    (weight_sigma**2 + self.weight_mu**2) / (2 * self.prior_sigma**2) - 0.5
                    
        kl_bias = torch.log(self.prior_sigma / bias_sigma) + \
                  (bias_sigma**2 + self.bias_mu**2) / (2 * self.prior_sigma**2) - 0.5
                  
        return kl_weight.sum() + kl_bias.sum()
