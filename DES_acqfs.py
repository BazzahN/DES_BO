'''
This script contains all of the cost aware acqfs and helper methods to be used in DES BO.
This includes:
-------------
- AEI
- SEI
- NEI
'''
import torch
from torch import Tensor
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils.probability.utils import (
    ndtr as Phi,
    phi,
)
from botorch.models.model import Model
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)

def _scaled_improvement(
    mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (mean - best_f) / sigma
    return u if maximize else -u


def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)

def _noise_var_penalty(sigma_2_eps,sigma_2_f,n):
    '''
    Calculates EI penalty: 
        (1 - sigma_eps/(sigma^2_eps + sigma^2_f * n)^(1/2))
    
    This penalises points with high noise
    '''
    ratio = sigma_2_eps.sqrt()/torch.sqrt(sigma_2_eps + sigma_2_f*n)

    return 1 - ratio


class DES_EI(AnalyticAcquisitionFunction):

    def __init__(
        self,
        model_f, #Main Model
        model_eps, #Noise Model 
        best_f,
        cost_model, # Cost model - we use linear cost model usually
        posterior_transform = None,
        maximize: bool = True,
    ):
        r"""Single-outcome Analytical Discrete Event Simulation Expected Improvement (analytic).

            alpha_{DES_EI}(x,n) = alpha_{EI(x)} * (\sigma_eps/(\sigma_eps^2 + \sigma_f^2 * n)^(1/2))

            Taylors EI for optimising discrete event simulation by incorperating heteroscedastic noise and the 
            ability to choose number of replications. 
        Args:
            model_f: A fitted single-outcome model.
            model_eps: A fitted single-outcome noise model
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            cost_model: Used to penalise costly decisions.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model_f, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.cost_model = cost_model
        self.maximize = maximize
        self.model_eps = model_eps
            
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate DES Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        self.to(device=X.device)  # ensures buffers / parameters are on the same device
        # print('X in shape \n')
        # print(X.shape)

        #TODO Implement code to account for unknown dimensions.
        N = X[...,-1].flatten() #Assumes n input is the extra dimension
        X_in = X[...,:-1]
        # print('\n N shape \n')
        # print(N.shape)

        # print('\n X shape \n')
        # print(X_in.shape)
        #Calculate Posterior of noise model for variance predictions
        posterior_eps = self.model_eps.posterior(
            X=X_in, posterior_transform=self.posterior_transform, observation_noise= False,
        )

        #Calculate predicted variance \sigma_eps^2
        sigma_2_eps = posterior_eps.mean.squeeze(-2).squeeze(-1) 

        #Calculates posterior for latent function f
        posterior_f = self.model.posterior(
            X=X_in, posterior_transform=self.posterior_transform, observation_noise= False,
        )
        # Calculate predicted f and \sigma_f^2
        mean = posterior_f.mean.squeeze(-2).squeeze(-1) 
        # print('mean shape is')
        # print(mean.shape)
        sigma_2_f = posterior_f.variance.clamp_min(1e-12).view(mean.shape)

        u = _scaled_improvement(mean, sigma_2_f.sqrt(), self.best_f, self.maximize)

        #Calculate penalty 
        penalty = _noise_var_penalty(sigma_2_eps,sigma_2_f,N)
        #Calculate query cost
        query_cost = self.cost_model(N)

        return (sigma_2_f.sqrt() * _ei_helper(u) * penalty * query_cost).squeeze(-1)

class AEI_fq(AnalyticAcquisitionFunction):
    r"""Single-outcome conservative posterior mean i.e.
    f_q(x) = f(x) - \sigma_f(x)

    Only supports the case of q=1. Requires the model's posterior to have a
    `mean` and 'sigma' property. The model must be single-outcome.

    Used for AEI when finding the conservative best solution i.e.
        f^* = \max f(x) - \sigma_f(x)

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PM = PosteriorMean(model)
        >>> pm = PM(test_X)
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome conservative posterior mean

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem. Note
                that if `maximize=False`, the posterior mean is negated. As a
                consequence `optimize_acqf(PosteriorMean(gp, maximize=False))`
                actually returns -1 * minimum of the posterior mean.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Posterior Mean values at the
            given design points `X`.
        """
        self.to(device=X.device)
        mean, sigma = self._mean_and_sigma(X)  # (b1 x ... x bk) x 1
        f_q = mean - sigma
        if not self.maximize:
            f_q = -f_q
        return f_q.squeeze(-1)
    
