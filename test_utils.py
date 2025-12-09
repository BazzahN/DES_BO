'''
Script contains all of the funcitons and methods for running the 1 dim heteroscedastic 
optimiation problem for the preliminary experiments with DES_BO 

Including
---------
- test_function
- noise_function
- LinearCostModel

'''
import torch
import numpy as np
from torch import Tensor

def test_function(x):
    '''
    Simple optimisation problem for DES_BO:
        f(x) = sin(5x) + cos(7x)

    Inputs
    ------
    x:Tensor
        The test/train locations for the GP
    Returns
    -------
    f: Float
        Evaluation of the test function
    '''

    return (torch.sin(5*x) + torch.cos(7*x)) 


def test_function_2(x):
    '''
    Simple optimisation problem to avoid edge problems for DES_BO:
    f(x) = sin4\pi*x + x
    '''

    return torch.sin(4*np.pi*x) + x


def heteroscedastic_noise(x,sigma2_0,phi = 0):
    '''
    Plots the underlying heteroscedastic noise surface for the test problem,
    defined as:
    \sigma_eps(x) = (cos(2 * \pi * x)/2 + 1) * sigma2_0

    Setting phi to \phi = 1.5 brings the peak varince to the centre
    Inputs
    ------
    x: tensor
        The test/train locations for the GP. Here x \in [0,1]
    sigma_0: float
        Scale of the heteroscedastic noise function
    phi: float
        shift paramater to move area of concentrated variance
    '''
    x = x - phi
    return (0.5 * torch.cos(2*torch.pi*x) + 1)* sigma2_0

def flat_noise(x,sigma2_0,phi=0):
    '''
    Plots the underlying homoscedastic noise surface for the test problem,
    defined as:
    \sigma_eps(x) = sigma2_0

    Inputs
    ------
    x: tensor
        The test/train locations for the GP. Here x \in [0,1]
    sigma_0: float
        Scale of the heteroscedastic noise function
    '''
    return sigma2_0 *torch.ones_like(x)

from botorch.models.deterministic import DeterministicModel

class InverseLinearCostModel(DeterministicModel):
    '''
    Deterministic, linear cost model for cost of evaluating DES at n replications:

        1/c(n) = 1/(a*n + b)
        
    where a and b are constants - these are supplied as arguments.
    '''

    def __init__(
            self,
            lin_coeffs): #list
        
        super().__init__()

        self.lin_coeffs= lin_coeffs
        self._num_outputs = 1

    def forward(self, N: Tensor) -> Tensor:
        '''
        Inputs
        ------
        N: Tensor 1xk
            Number of replications to be used at evaluation point
        
        Returns
        -------
        c: Tensor
            Cost evaluation at points in n
        
        '''
        cost = 1/(self.lin_coeffs[0]*N + self.lin_coeffs[1])
        return cost

#TODO Linear Regression linear cost model
'''
Currently cost model uses coefficients chosen by me. In actual implementation these will 
have to be determined. IT is quite easy to do this but modifications are needed:
    - DES code should output its execution time to be fed into the code
    - 
'''