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
tkwargs = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}
def test_function_1(x):
    '''
    Simple Sinosodial function. Used in inital BODES experiments:
        f(x) = sin(5x) + cos(7x)
    
    Domain: x \in [0,1]

    Inputs
    ------
    x:Tensor
        The test/train locations for the GP

    Returns
    -------
    f: Float
        Evaluation of the test function
    '''

    return torch.sin(5*x) + torch.cos(7*x)

def test_function_2(x):
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

    return -(torch.sin(5*x) + torch.cos(7*x)) 

# def test_function_2(x):
#     '''
#     Simple optimisation problem to avoid edge problems for DES_BO:
#     f(x) = sin4\pi*x + x
#     '''

#     return torch.sin(4*np.pi*x) + x


def noise_function_1(x,phi=0,tau=1):
    '''
    Plots the underlying heteroscedastic noise surface for the test problem,
    defined as:
    \sigma_eps^2(x) = (0.1 + tau*exp(-0.5*((x - \phi)/0.4)**2))**2
    
    Describes noise as a sinlge peak centred at phi with a constant
    background noise of 0.1 everywhere else. 

    Domain: x \in [-\infty,infty]
    Inputs
    ------
    x: tensor
        The test/train locations for the GP. Here x \in [0,1]
    
    phi: float: default =0
        Shifting paramater. Default centred at 0
    tau: float: default =1
        Scale of the heteroscedastic noise function. Chamges peak to be tau + 0.1. 
        Default settings have variance of 1 + 0.1 at the centre
    '''
    x = x - phi

    #TODO Turn c into a tunable paramater
    '''
    c lets us change the width of the noise bump. Previously this was c=0.4, but c=0.1 makes more sense
    for the original test function. 
    '''
    c = 0.1
    return (0.3 + tau*torch.exp(-0.5*(x/c)**2))**2

def noise_function_2(x,phi=0,tau=1):
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
    return tau *torch.ones_like(x)

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

class Target_Function:

    def __init__(self,
                 test_function,
                 noise_function,
                 phi,
                 tau,
                 rng_state):
        
        self.test_function = test_function
        self.noise_function = noise_function
        self.phi = phi
        self.tau = tau
        self.rng = torch.Generator().set_state(rng_state)

    def get_rng_state(self):

        return self.rng.get_state()
    def update_rng_state(self,rng_state):

        self.rng = self.rng.set_state(rng_state)
    
    def eval_target_noisy(self,test_x,test_n,moments=1):
        '''
            Generates noisy evaluations by generating n x k noisy
            y values. 
            
            :param self: Description
            :param test_x: Description
            :param test_n: Description
            :param moments: int default = 1
            
            Returns
            ------
            test_x_expand: n x k
            y_evals: n x k nparray
            sigma_2: k sized nparray
            '''
        
        n = int(test_n[0].item())
        # Calculate sample standard deviation
        sigma_2 = self.noise_function(test_x,self.phi,self.tau)
        s_vec = torch.sqrt(sigma_2)

        #Duplicate 
        s_vec_expand = torch.tile(s_vec.squeeze(),(n,1))
        test_x_expand = torch.tile(test_x.squeeze(),(n,1))

        #Generate noisy evaluations
        noise_std = torch.normal(mean=0,
                                 std=1,
                                 size=test_x_expand.shape,
                                 generator=self.rng) * s_vec_expand        
        
        y_evals = self.test_function(test_x_expand) + noise_std

        #If moments is true output aggregated data
        '''
        If moments =True then output the k length data set of locations, sample means
        and sample variances.

        If moments = False then outptu the kxn length data set of locations and samples.
        '''
        if moments:
            y_out = y_evals.mean(0).unsqueeze(1)
            sigma2_out = y_evals.var(0).unsqueeze(1)

            return test_x, y_out,sigma2_out
        else:
            return test_x_expand, y_evals,0
    
    def eval_target_true(self,test_x):

        f_out = self.test_function(test_x).to(**tkwargs) 
        sigma2_out = self.noise_function(test_x,self.phi,self.tau).to(**tkwargs)

        return f_out, sigma2_out
    
    def eval_target_true_grid(self,n_grid,x_min,x_max):
        
        test_x = torch.linspace(x_min,x_max,n_grid).reshape(n_grid,1).to(**tkwargs)
        f_out = self.test_function(test_x).to(**tkwargs) 
        sigma2_out = self.noise_function(test_x,self.phi,self.tau).to(**tkwargs)

        return test_x,f_out, sigma2_out


def get_k_inital_evals(k,n,target_function,x_min,x_max):
    '''
    Gets k inital observations for a flat n replications each  

    Inputs
    ------
    k: int,
        Number of evaluation points
    n: int
        Number of replications taken at each point
    target_function: Target_Function Object
        The initiallised Target_Function Class Object
    x_min: float
        Minimum bound of test function domain
    x_max: float
        Maximum bound of the test function domain
    '''

    #Generate k equally spaced training points
    train_x = torch.linspace(x_min,x_max,k).reshape(k,1).to(**tkwargs)
    train_n = torch.ones_like(train_x) * n

    #Generate y values from latent function plus heteroscedastic Gaussian noise
    train_x, train_y, train_sigma2 = target_function.eval_target_noisy(train_x,train_n)

    return train_x,train_n,train_y,train_sigma2, target_function


#TODO Linear Regression linear cost model
'''
Currently cost model uses coefficients chosen by me. In actual implementation these will 
have to be determined. IT is quite easy to do this but modifications are needed:
    - DES code should output its execution time to be fed into the code
    - 
'''
TEST_FUNCTION_DIAL = [test_function_1,
                      test_function_2]

NOISE_FUNCTION_DIAL =[noise_function_1,
                      noise_function_2]


