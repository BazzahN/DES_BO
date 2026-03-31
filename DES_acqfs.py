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
from botorch.exceptions.warnings import legacy_ei_numerics_warning
from botorch.utils.probability.utils import (
    ndtr as Phi,
    phi,
)
from botorch.models.model import Model
from HGP_utils import HeteroscedasticBOModel
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)

#Underscores are a convention to denote internal use of functions
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
    ratio = sigma_2_eps/(sigma_2_eps + sigma_2_f*n)

    return 1 - ratio.sqrt()


def _inverse_log_transform(eps_log,eps_v_log,output_transform):

    return output_transform.inv_log_transform(eps_log,eps_v_log)

def _transform_GP(f,f_var,output_transform):
    
    #Unstandardise zeta(f)-> f
    var_transform = f_var * output_transform.sig_std
    mu_transform = output_transform.unstandardise(f)

    return mu_transform,var_transform


def _model_type(model):
    """
    Function to ensure compatability with acquisition funciton.
    This exposes the correct model and posterior based upon
    the nature of the model used. If VIHGP then the model and 
    its posterior are exposed. If SKHGP
    """
    if isinstance(model,HeteroscedasticBOModel):
        return model,model.posterior, model.noise_posterior 
    else:
        return model['f'], model['f'].posterior, model['eps'].posterior


class DES_EI(AnalyticAcquisitionFunction):

    def __init__(
        self,
        model, #Now takes model dictionary instead of single model
        output_transform, #Unstandardise GP output
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
        model_f, model_f_posterior, model_eps_posterior = _model_type(model)

        super().__init__(model=model_f, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.output_transform = output_transform
        self.cost_model = cost_model
        self.maximize = maximize
        # self.model_eps = model['eps']

        #Assign Posteriors
        self.model_eps_posterior = model_eps_posterior
        self.model_f_posterior = model_f_posterior

    @t_batch_mode_transform(expected_q=1)        
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
      
        #TODO Implement code to account for unknown dimensions.
        N = X[...,-1] # Assumes n input is the extra dimension
        X_in = X[...,:-1]

        # To keep botorch-compatible t-batch dimensions (k,1,1,1), push a singleton q dim
        X_eval = X_in.unsqueeze(-3)

        # Calculate Posterior of noise model for variance predictions
        posterior_eps = self.model_eps_posterior(
            X=X_eval, posterior_transform=self.posterior_transform, observation_noise=False,
        )

        # Calculate predicted variance \sigma_eps^2
        sigma_2_eps = posterior_eps.mean.squeeze(-1).squeeze(-1)
        sigma_2_eps_var = posterior_eps.variance.clamp_min(1e-12).view_as(sigma_2_eps)

        ##Unstandardise Noise GP preds
        #NOTE Dirty, Evil Code
        if type(self.output_transform) is dict:
            sigma_2_eps,sigma_2_eps_var = _transform_GP(sigma_2_eps,sigma_2_eps_var,self.output_transform['eps'])
            sigma_2_eps = _inverse_log_transform(sigma_2_eps,sigma_2_eps_var,self.output_transform['eps']) * self.output_transform['f'].sig_std
        else:
            sigma_2_eps = _inverse_log_transform(sigma_2_eps,sigma_2_eps_var,self.output_transform) * self.output_transform.sig_std
        
        # Calculates posterior for latent function f
        posterior_f = self.model_f_posterior(
            X=X_eval, posterior_transform=self.posterior_transform, observation_noise=False,
        )

        # Calculate predicted f and \sigma_f^2
        mean = posterior_f.mean.squeeze(-1).squeeze(-1)
        sigma_2_f = posterior_f.variance.clamp_min(1e-12).view_as(mean)
  
        ##Unstandardise Signal GP preds
        if type(self.output_transform) is dict:
            mean,sigma_2_f = _transform_GP(mean,sigma_2_f,self.output_transform['f'])
        else:
            mean,sigma_2_f = _transform_GP(mean,sigma_2_f,self.output_transform)
            
        #Calculate Augmented Expected Improvement
        u = _scaled_improvement(mean, sigma_2_f.sqrt(), self.best_f, self.maximize)

  
        ##Calculate penalty 
        penalty = _noise_var_penalty(sigma_2_eps,sigma_2_f,N)
     
        ##Calculate query cost
        query_cost = self.cost_model(N.flatten())
        out = (sigma_2_f.sqrt() * _ei_helper(u) * penalty).squeeze(-1) * query_cost

        return out

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
        output_transform, #Used to unstandardise GP preds
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
        self.output_transform = output_transform

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
        
        #Transfomr mean and sigma GP predictions
        if type(self.output_transform) is dict:
            mean,sigma = _transform_GP(mean,sigma,self.output_transform['f'])
        else:
            mean,sigma = _transform_GP(mean,sigma,self.output_transform)
       
        f_q = mean - sigma
        if not self.maximize:
            f_q = -f_q
        return f_q.squeeze(-1)
    


from botorch.acquisition import MaxValueBase


from botorch.models.utils import check_no_nans


#Variance Lowerbound - avoids tiny variance
CLAMP_LB = 1.0e-8

class BODES_IG(MaxValueBase):
    r"""The acquisition function for General-purpose Information-Based
    Bayesian Optimisation (GIBBON).

    This acquisition function provides a computationally cheap approximation of
    the mutual information between max values and a batch of candidate points `X`.
    See [Moss2021gibbon]_ for a detailed discussion.

    The model must be single-outcome, unless using a PosteriorTransform.
    q > 1 is supported through greedy batch filling.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> qGIBBON = qLowerBoundMaxValueEntropy(model, candidate_set)
        >>> candidates, _ = optimize_acqf(qGIBBON, bounds, q=5)
    """

    def __init__(
        self,
        model, #NOTE:CHANGE: Now takes a dict: the stochastic kriging model 
        cost_model, #Linear Cost Model
        output_transform, #Unstandardise GP output
        candidate_set: Tensor,
        num_mv_samples: int = 10,
        posterior_transform: PosteriorTransform | None = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Tensor | None = None,
        train_inputs: Tensor | None = None,
    ) -> None:
        r"""Lower bound max-value entropy search acquisition function (GIBBON).

        Args:
            model: dict
                A dictionary containing the latent model 'f' and the noise model 'eps'
            candidate_set: A `n x d` Tensor including `n` candidate points to
                discretize the design space. Max values are sampled from the
                (joint) model posterior over these points.
            num_mv_samples: Number of max value samples.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            use_gumbel: If True, use Gumbel approximation to sample the max values.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            train_inputs: A `n_train x d` Tensor that the model has been fitted on.
                Not required if the model is an instance of a GPyTorch ExactGP model.
        """

        model_f, model_f_posterior, model_eps_posterior = _model_type(model)
        super().__init__(
            model=model_f, #NOTE:CHANGE: send only the latent model to MaxValueBase
            candidate_set=candidate_set,
            num_mv_samples=num_mv_samples,
            posterior_transform=posterior_transform,
            use_gumbel=use_gumbel,
            maximize=maximize,
            X_pending=X_pending,
            train_inputs=train_inputs,
        )
        #Assign Posteriors
        self.model_eps_posterior = model_eps_posterior
        self.model_f_posterior = model_f_posterior

        self.output_transform = output_transform
        self.cost_model = cost_model
        self.set_X_pending(X_pending)

    #NOTE:CHANGE: Override forward method 

    @t_batch_mode_transform(expected_q=1)
    # @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute max-value entropy at the design points `X`.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of MVE values at the given design points `X`.
        """
        ##Marshall input
        N = X[...,-1] #shape [k,1]
        X_in = X[...,:-1] #shape [k,1,1]

        # Compute the posterior of both noise and latent model
        posterior_f = self.model_f_posterior(
            X=X_in.unsqueeze(-3),
            observation_noise=False,
            posterior_transform=self.posterior_transform,
        )
        posterior_eps = self.model_eps_posterior(
            X=X_in.unsqueeze(-3),#make [k,1,1,1]
            observation_noise=False,
            posterior_transform=self.posterior_transform,

        )
        #Calculate predicted variance \sigma_eps^2
        
        sigma_2_eps = posterior_eps.mean.squeeze(-1).squeeze(-1)  # make [k,1]
        sigma_2_eps_var = posterior_eps.variance.clamp_min(CLAMP_LB).view_as(sigma_2_eps)
        
        ## Transform predicted variance
        if type(self.output_transform) is dict:
            
            sigma_2_eps, sigma_2_eps_var = _transform_GP(
                sigma_2_eps, sigma_2_eps_var, self.output_transform['eps']
            )
            sigma_2_eps = (
                _inverse_log_transform(
                    sigma_2_eps, sigma_2_eps_var, self.output_transform['eps']
                )
                * self.output_transform['f'].sig_std
            )
        else:
            sigma_2_eps = (
                _inverse_log_transform(sigma_2_eps, sigma_2_eps_var, self.output_transform)
                * self.output_transform.sig_std
            )

        # Calculate predicted mean
        mean_f = posterior_f.mean.squeeze(-1).squeeze(-1)
        sigma_2_f = posterior_f.variance.clamp_min(CLAMP_LB).view_as(mean_f)
        
        ##transform predicted mean
        
        if type(self.output_transform) is dict:
            mean_f,sigma_2_f = _transform_GP(mean_f,sigma_2_f,self.output_transform['f'])
        else:
            mean_f,sigma_2_f = _transform_GP(mean_f,sigma_2_f,self.output_transform)
        
        sigma_f = sigma_2_f.sqrt()

        
        # Average over fantasies, ig is of shape `num_fantasies x batch_shape x (m)`.
        
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # prepare max value quantities required by GIBBON
        mvs = torch.transpose(self.posterior_max_values, 0, 1)
        
        ##Transform max value quantities
        if type(self.output_transform) is dict:
            mvs,_ = _transform_GP(mvs,mvs,self.output_transform['f'])
        else:
            mvs,_ = _transform_GP(mvs,mvs,self.output_transform)
        
        # print("mvs shap is",mvs.shape)
        # print("mean_f shap is",mean_f.shape)
  
        # 1 x s_M
        normalized_mvs = (mvs - mean_f) / sigma_f
        # batch_shape x s_M

        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))
        ratio = pdf_mvs / cdf_mvs
        check_no_nans(ratio)

        # prepare squared correlation between current and target fidelity
        rho_sq = N * sigma_2_f / (sigma_2_eps + N * sigma_2_f)

        # batch_shape x 1
        check_no_nans(rho_sq)

        # calculate quality contribution to the GIBBON acquisition function
        inner_term = 1 - rho_sq * ratio * (normalized_mvs + ratio)
        # print(f'The inner term is {rho}')
        acq = -0.5 * inner_term.clamp_min(CLAMP_LB).log()
        # average over posterior max samples
        costs = self.cost_model(N.squeeze(-1))
        acq = acq.mean(dim=1) * costs
        
        #Average over fantasies
        # acq = acq.mean(dim=0)
        return acq #shape out [k]

    def _compute_information_gain(
        self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor
    ) -> Tensor:
        r"""Compute GIBBON's approximation of information gain at the design points `X`.

        When using GIBBON for batch optimization (i.e `q > 1`), we calculate the
        additional information provided by adding a new candidate point to the current
        batch of design points (`X_pending`), rather than calculating the information
        provided by the whole batch. This allows a modest computational saving.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design point each.
            mean_M: A `batch_shape x 1`-dim Tensor of means.
            variance_M: A `batch_shape x 1`-dim Tensor of variances
                consisting of `batch_shape` t-batches with `num_fantasies` fantasies.
            covar_mM: A `batch_shape x num_fantasies x (1 + num_trace_observations)`
                -dim Tensor of covariances.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of information gains at the
            given design points `X`.
        """
       
        return print('no never')