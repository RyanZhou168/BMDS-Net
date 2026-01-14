"""
Bayesian Layers Module.
Implements Variational Inference for Convolutional Layers (Bayes By Backprop).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianConv3d(nn.Module):
    """
    Bayesian Conv3d Layer with Variational Inference.
    Replaces the standard deterministic Conv3d layer.
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=1, 
                 stride=1, 
                 padding=0, 
                 prior_mean=0, 
                 prior_variance=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Learnable Parameters: Mean (mu) and Log Variance (logvar)
        # Shape: [out, in, k, k, k]
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_channels))
        
        # Fixed Prior (Gaussian)
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.prior_logvar = torch.log(torch.tensor(prior_variance))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize Mu with Kaiming (Standard initialization)
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias_mu, 0)
        
        # Initialize LogVar to a small value (low initial uncertainty)
        # exp(-5) is very small, ensuring stability at the start of fine-tuning
        nn.init.constant_(self.weight_logvar, -5) 
        nn.init.constant_(self.bias_logvar, -5)
    
    def forward(self, x):
        if self.training:
            # Reparameterization Trick: w = mu + sigma * epsilon
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight_epsilon = torch.randn_like(weight_std)
            weight = self.weight_mu + weight_std * weight_epsilon
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_epsilon = torch.randn_like(bias_std)
            bias = self.bias_mu + bias_std * bias_epsilon
        else:
            # During deterministic inference, use the mean weights
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.conv3d(x, weight, bias, self.stride, self.padding)
    
    def kl_divergence(self):
        """
        Calculate KL Divergence: KL(q(w|theta) || p(w))
        Analytical solution for two Gaussians.
        """
        kl_weight = -0.5 * self.weight_logvar + 0.5 * (self.prior_logvar - self.prior_mean + (self.weight_mu**2 + torch.exp(self.weight_logvar)) / self.prior_variance)
        kl_bias = -0.5 * self.bias_logvar + 0.5 * (self.prior_logvar - self.prior_mean + (self.bias_mu**2 + torch.exp(self.bias_logvar)) / self.prior_variance)
        return kl_weight.sum() + kl_bias.sum()
