import torch 
from torch import Tensor

from botorch.acquisition import MaxValueBase
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.models.utils import check_no_nans

from linear_operator.functions import inv_quad


from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)


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

        
        super().__init__(
            model=model['f'], #NOTE:CHANGE: send only the latent model to MaxValueBase
            candidate_set=candidate_set,
            num_mv_samples=num_mv_samples,
            posterior_transform=posterior_transform,
            use_gumbel=use_gumbel,
            maximize=maximize,
            X_pending=X_pending,
            train_inputs=train_inputs,
        )
        self.model_eps = model['eps']
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
        posterior_f = self.model.posterior(
            X=X_in.unsqueeze(-3),
            observation_noise=False,
            posterior_transform=self.posterior_transform,
        )
        posterior_eps = self.model_eps.posterior(
            X=X_in.unsqueeze(-3),#make [k,1,1,1]
            observation_noise=False,
            posterior_transform=self.posterior_transform,

        )

        # #Calculate predicted variance \sigma_eps^2
        sigma_2_eps = posterior_eps.mean.squeeze(-1).squeeze(-1) # make [k,1]

        mean_f = posterior_f.mean.view_as(sigma_2_eps)
        sigma_2_f = posterior_f.variance.clamp_min(CLAMP_LB).view_as(mean_f)
        sigma_f = sigma_2_f.sqrt()
        # Average over fantasies, ig is of shape `num_fantasies x batch_shape x (m)`.
        
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # prepare max value quantities required by GIBBON
        mvs = torch.transpose(self.posterior_max_values, 0, 1)
        #print(f'The sampled max values are:\n {mvs}\n')
        # 1 x s_M
        normalized_mvs = (mvs - mean_f) / sigma_f
        # batch_shape x s_M

        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))
        ratio = pdf_mvs / cdf_mvs
        check_no_nans(ratio)

        # prepare squared correlation between current and target fidelity
        rho_sq = N * sigma_2_f / (sigma_2_eps + N * sigma_2_f)
        # print(f'The rho is {rho}')
        # batch_shape x 1
        check_no_nans(rho_sq)

        # calculate quality contribution to the GIBBON acquisition function
        inner_term = 1 - rho_sq * ratio * (normalized_mvs + ratio)
        # print(f'The inner term is {rho}')
        acq = -0.5 * inner_term.clamp_min(CLAMP_LB).log()
        # average over posterior max samples
        costs = self.cost_model(N.squeeze(-1))
        # print(f'the costs are {costs}')
        # print(f'The final acq value is {acq}')
        acq = acq.mean(dim=1)*costs
        
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