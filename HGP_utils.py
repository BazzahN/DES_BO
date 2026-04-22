# het_joint_gp.py
import math
import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
#from gpytorch.lazy import DiagLazyTensor
#from gpytorch.mlls.variational_elbo import _approx_log_normal_cdf  # not used but keep gpytorch import style
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition.objective import PosteriorTransform
#from botorch.models.model import Model
from torch.nn import Module

from botorch.models.utils.gpytorch_modules import (get_covar_module_with_dim_scaled_prior
                                                   ,get_matern_kernel_with_gamma_prior)


# ------------------------------------------------------------------
# 1) Joint latent variational GP (2 latent GPs: f and g)
# ------------------------------------------------------------------
class HeteroscedasticLatentGP(gpytorch.models.ApproximateGP):
    """
    Multitask variational GP with 2 independent latent processes (f and g).
    The variational distribution is batch-shaped for the latents.
    
    By default this defines two independent Gaussian Processes
    each with the same type of mean function and kernel function. 
    The paramaters are different. 
    
    """
    def __init__(self, 
                 inducing_points,
                 covar_module = None,
                 mean_module = None,):
        # inducing_points: (m, d) -> gpytorch expects (m, d)
        num_latents = 2
        num_inducing = inducing_points.size(-2)

        # Variational distribution: one variational distribution per latent (batch_shape = [num_latents])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([num_latents])
        )

        # Wrap in VariationalStrategy then IndependentMultitaskVariationalStrategy
        base_variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base_variational_strategy,
            num_tasks=num_latents
        )

        super().__init__(variational_strategy)

        # Batch-shaped mean and kernel (one for each latent)
        if covar_module is None:
            # covar_module = gpytorch.kernels.ScaleKernel(
            #     gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            #     batch_shape=torch.Size([num_latents])).to(inducing_points)

            # covar_module = get_covar_module_with_dim_scaled_prior(ard_num_dims=1,batch_shape=torch.Size([num_latents])).to(inducing_points)
            covar_module = get_matern_kernel_with_gamma_prior(ard_num_dims=1,batch_shape=torch.Size([num_latents])).to(inducing_points)              
        
        if mean_module is None:
            #NOTE: I had neglected to note down that I had zero mean.
            # mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents])).to(inducing_points)
            mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents])).to(inducing_points)
        
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.to(inducing_points) #Ensures that chosen data type is preserved througout the model

    def forward(self, x):
        # x: (n, d)
        mean = self.mean_module(x)                  # shape: (num_latents, n)
        cov = self.covar_module(x)                  # returns a LazyTensor with batch_shape [num_latents]
        return MultivariateNormal(mean, cov)


# ------------------------------------------------------------------
# 2) Heteroscedastic likelihood overriding expected_log_prob
# ------------------------------------------------------------------
import numpy as np
class HeteroscedasticGaussianLikelihood(gpytorch.likelihoods.Likelihood):
    """
    Likelihood: y | f, g ~ N(f, exp(g))
    We implement expected_log_prob(target, function_dist, num_samples=1)
    which should return a tensor of shape (num_samples,) containing the
    log probability sums for each sample. The VariationalELBO will average
    over the samples.
    """
    def __init__(self):
        super().__init__()

    def expected_log_prob(self, target, function_dist, num_samples=1):
        """
        target: (n,) or (batch, n) - here we assume simple (n,)
        function_dist: a gpytorch.distributions.MultivariateNormal with
                       batch_shape = [num_latents] and event_shape = [n],
                       OR with sample_shape if sampled.
        num_samples: number of MC samples to estimate the expectation.
        Returns:
            tensor of shape (num_samples,) where each entry is the sum of log probs over datapoints.
        """

        #NOTE Coded out as we calculate the expected log prob directly from mean
        # Draw reparameterized samples from q(f,g):
        # samples shape -> (num_samples, num_latents, n)

        # samples = function_dist.rsample(sample_shape=torch.Size([num_samples]))
        # # Ensure shapes: samples[..., latent_idx, datapoint]
        # # Extract f and g:
        # f_samps = samples[..., 0, :]   # (num_samples, n)
        # g_samps = samples[..., 1, :]   # (num_samples, n)

        mean = function_dist.mean
        var = function_dist.variance

        f_mean = mean[..., 0]
        g_mean = mean[..., 1]

        sigma2_f = var[..., 0]
        g_var = var[..., 1]

        target = target.squeeze(-1)

        # Compute noise per-sample (positive)
        #noise_var = torch.exp(g_samps)  # (num_samples, n)

        # residuals
        # target shape might be (n,) -> expand to (num_samples, n)
        # if target.dim() == 1:
        #     target_exp = target.unsqueeze(0).expand(num_samples, -1)
        # else:
        #     # fallback: allow batched targets (num_targets, n) - but keep simple here
        #     target_exp = target

        # res = target_exp - f_samps  # (num_samples, n)

        # Normal log-prob per datapoint per sample
        # log p = -0.5 * (log(2*pi*noise) + res^2 / noise)

        sigma2_eps_inv = torch.exp(-g_mean + 0.5*g_var)

        sqr_term = (target - f_mean) ** 2

        log_prob = -0.5 * (np.log(2.0 * math.pi) + g_mean + (sqr_term + sigma2_f) * sigma2_eps_inv)

        # Sum over datapoints -> (num_samples,)
        log_prob_sum = log_prob.sum(dim=-1)

        return log_prob_sum
    
    ##NOTE FORWARD METHOD DOES NOTHING. IT EXISTS TO MAKE LIKELIHOOD CLASS INHERITENCE WORK
    def forward(self, function_samples, **kwargs):
        """
        function_samples shape:
        sample_shape x N x 2
        """
        f = function_samples[..., 0]
        g = function_samples[..., 1]

        noise = torch.exp(g)

        return torch.distributions.Normal(f, noise.sqrt())

'''
Calculates ELBO in the same manner as GPJax, using moments. I believe this is based on the lower bound provided by 
Lavello paper
'''

class HeteroscedasticELBO(gpytorch.mlls.MarginalLogLikelihood):
    """
    Minimal ELBO: E_q[log p(y|f,g)] - KL[q(u) || p(u)]
    We override forward to accept latent_dist (q(f,g)) and target
    """

    def __init__(self, likelihood, model):
        # We subclass MarginalLogLikelihood for API parity, but we implement a Variational ELBO
        super().__init__(likelihood, model)
    def forward(self, latent_dist, target):
        """
        latent_dist: MultivariateNormal with batch_shape=[num_latents] event_shape=[n]
        target: (n,)
        Returns: scalar ELBO (not negated)
        """
        # Expected log likelihood (MC)
        exp_log_prob = self.likelihood.expected_log_prob(target, latent_dist)
     
        # KL from variational strategy (this sums over all inducing points and latents)
        # For the IndependentMultitaskVariationalStrategy, kl_divergence() is provided
        kl_div = self.model.variational_strategy.kl_divergence()

        # ELBO
        elbo = exp_log_prob -  kl_div

        return elbo


from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition.objective import PosteriorTransform
from torch.nn import Module

class HeteroscedasticBOModel(GPyTorchModel):
    """
    Wrapper around the latent GP to expose posterior(X) returning a GPyTorchPosterior
    that BoTorch acquisition functions can consume.
    
    """
    def __init__(self, 
                 train_x,
                 train_y,
                 inducing_points,
                 likelihood=None,
                 mean_module=None,
                 covar_module=None,):
        """
        Args:
            train_X: Training inputs (due to the ability of the SVGP to sub-sample
                this does not have to be all of the training inputs).
            train_Y: Training targets (optional).
            likelihood: Instance of a GPyTorch likelihood. If omitted, uses a
                either a ``GaussianLikelihood`` (if ``num_outputs=1``) or a
                ``MultitaskGaussianLikelihood``(if ``num_outputs>1``).
            covar_module: Kernel function. If omitted, uses an ``RBFKernel``.
            mean_module: Mean of GP model. If omitted, uses a ``ConstantMean``.
            inducing_points: The number or specific locations of the inducing points.
          
        """
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.model = HeteroscedasticLatentGP(inducing_points,
                                             covar_module,
                                             mean_module)

        if likelihood is None:
            self.likelihood = HeteroscedasticGaussianLikelihood() 
        else:
             self.likelihood = likelihood
 
        self._desired_num_outputs = 1 #NOTE Forced to change this to 1 was >1 ouptuts are not supported by the botorch acquisition functions I am using. 

    @property
    def num_outputs(self):
        return self._desired_num_outputs

 
    def eval(self):
            r"""Puts the model in ``eval`` mode."""
            return Module.eval(self)

    
    def train(self, mode: bool = True):
            r"""Put the model in ``train`` mode.

            Args:
                mode: A boolean denoting whether to put in ``train`` or ``eval`` mode.
                    If ``False``, model is put in ``eval`` mode.
            """
            return Module.train(self, mode=mode)

    def posterior(
            self,
            X,
            output_indices: list[int] | None = None,
            latent_model:   bool = True,
            observation_noise: bool = False,
            posterior_transform: PosteriorTransform | None = None,
        ) -> GPyTorchPosterior:
            if output_indices is not None:
                raise NotImplementedError(  # pragma: no cover
                    f"{self.__class__.__name__}.posterior does not support output indices."
                )
            self.eval()  # make sure model is in eval mode

            # input transforms are applied at ``posterior`` in ``eval`` mode, and at
            # ``model.forward()`` at the training time
            X = self.transform_inputs(X)

            # check for the multi-batch case for multi-outputs b/c this will throw
            # warnings


            """
            To implement: If lantent model is true this sets num_outputs to 1, then
            splits the dist variable and outputs a posterior for latent only. This
            is to be used by the acquisition functions down the line. 
            
            """

            X_ndim = X.ndim
            # if self.num_outputs > 1 and X_ndim > 2:
            #     X = X.unsqueeze(-3).repeat(*[1] * (X_ndim - 2), self.num_outputs, 1, 1)

            if X_ndim > 2:
                X = X.unsqueeze(-3).repeat(*[1] * (X_ndim - 2), 2, 1, 1)
            dist = self.model(X)

            #NOTE Should I ever need to use the observational distribution I will have to change
            #the Heteroscedastic Likelihood function to properly calculate noise. 
            if observation_noise:
                dist = self.likelihood(dist)

            #TODO Added to pass to max value estimation function
            if latent_model:
                idx = 0 #Latent function index.
                # Use batch-dimension indexing (not event-dimension) to preserve q/batch dims.
                dist = dist[...,idx]
            posterior = GPyTorchPosterior(distribution=dist)
            #NOTE Outcome and posterior transform is not and should not be used for my code.
            # All transformation is doen in the botorch code
            
            if hasattr(self, "outcome_transform"):
                posterior = self.outcome_transform.untransform_posterior(posterior, X=X)
            if posterior_transform is not None:
                posterior = posterior_transform(posterior=posterior, X=X)

            return posterior
    def noise_posterior(
            self,
            X,
            output_indices: list[int] | None = None,
            observation_noise: bool = False, #Does nothing if observational noise is switched on
            posterior_transform: PosteriorTransform | None = None,
        ) -> GPyTorchPosterior:
            if output_indices is not None:
                raise NotImplementedError(  # pragma: no cover
                    f"{self.__class__.__name__}.posterior does not support output indices."
                )
            self.eval()  # make sure model is in eval mode

            # input transforms are applied at ``posterior`` in ``eval`` mode, and at
            # ``model.forward()`` at the training time
            X = self.transform_inputs(X)

            # check for the multi-batch case for multi-outputs b/c this will throw
            # warnings

        
            X_ndim = X.ndim
            if X_ndim > 2:
                X = X.unsqueeze(-3).repeat(*[1] * (X_ndim - 2), 2, 1, 1)
            dist = self.model(X)

            #NOTE Should I ever need to use the observational distribution I will have to change
            #the Heteroscedastic Likelihood function to properly calculate noise. 

    
            # if observation_noise:
            #     dist = self.likelihood(dist)

            #TODO Added to pass to max value estimation function
            idx = 1 #noise function index.
            # Use batch-dimension indexing (not event-dimension) to preserve q/batch dims.
            dist = dist[...,idx]

            posterior = GPyTorchPosterior(distribution=dist)
            
            #NOTE Outcome and posterior transform is not and should not be used for my code.
            # All transformation is doen in the botorch code
            
            if hasattr(self, "outcome_transform"):
                posterior = self.outcome_transform.untransform_posterior(posterior, X=X)
            if posterior_transform is not None:
                posterior = posterior_transform(posterior=posterior, X=X)

            return posterior
    def forward(self, X) -> MultivariateNormal:
            if self.training:
                X = self.transform_inputs(X)
            return self.model(X)


def fit_vihgp_elbo(model,elbo,iters,verbose=False):
    """
    Given variational elbo object and specified number of epochs
    this function fits the variational hgp to the data.
    
    Args:
        elbo: ApproximateMarginalloglikelihood 
            Only accepts the custom built HeteroscedasticElBO
            gpytorch.mll object made for this problem
        iters: int 
            Number of iterations to run 
        verbose: bool
            Output negative elbo after 200 iters.
    """ 
    q_model = elbo.model#Extracts GPYtorch model
    train_x = model.train_x
    train_y = model.train_y

    q_model.train()
    optimiser=torch.optim.Adam(q_model.parameters(),lr=0.01)
    for i in range(iters):
        optimiser.zero_grad()
        q_distb = q_model(train_x)

        loss = -elbo(q_distb,train_y)
        loss.backward()
        optimiser.step()

        if (i+1) % 200 ==0 and verbose==1:
            print(f"Iter {i+1}/{iters} - negative ElBO {loss.item():.4f}")

    return model,elbo