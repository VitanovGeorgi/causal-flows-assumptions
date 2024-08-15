import pdb

import torch
import torch.distributions as distr

from causal_nf.distributions.heterogeneous import Heterogeneous
from torch.distributions import Independent, Normal, Uniform, Laplace, Bernoulli

pu_dict = {}

acceptable_distr = ['Normal', 'Uniform', 'Bernoulli', 'Laplace', 'MultivariateNormal', 'Custom']
distr_dict = {
    'normal': Normal,
    'uniform': Uniform,
    'laplace': Laplace,
    'multivariatenormal' : distr.multivariate_normal.MultivariateNormal,
    'bernoulli': Bernoulli
}


def create_multivariate_normal_dist(
        base_distribution_name: str = 'MultivariateNormal',
        means: list = None,
        variances: list = None,
        correlations: list = None,
        no_nodes: int = 5
) -> torch.distributions:
    if base_distribution_name.lower() != 'multivariatenormal':
        raise NameError()
    # pdb.set_trace()
    default_mean = torch.zeros(no_nodes)
    if means is not None:
        if not isinstance(means, list):
            raise ValueError(f"Means {means} needs to be a list(list).")
        
        for mean_pair in means:
            
            if not isinstance(mean_pair, list):
                raise ValueError(f"{mean_pair} needs to be a list.")
            
            if len(mean_pair) != 2:
                raise ValueError(f"{mean_pair} is not a valid input for means, it needs to be a list of only two elements: [node_number, mean]")
                continue
            
            if mean_pair[0] + 1 > no_nodes:
                pdb.set_trace()
                raise ValueError(f"{mean_pair[0]} is not a valid input, as there aren't as many nodes, {no_nodes}, in the graphs")
            
            if not isinstance(mean_pair[0], int) or mean_pair[0] < 0:
                raise ValueError(f"{mean_pair[0]} needs to be an int, and non-negative.")
            
            if not isinstance(mean_pair[1], (int, float)):
                raise ValueError(f"{mean_pair[1]} needs to be a (int, float).")

            default_mean[mean_pair[0]] = mean_pair[1]

    default_variance = torch.eye(no_nodes) # this is actually the cov_matrix
    if variances is not None:
         if not isinstance(variances, list):
            raise ValueError(f"{variances} needs to be a list(list).")
         
         for variances_pair in variances:
            
            if not isinstance(variances_pair, list):
                raise ValueError(f"{variances_pair} needs to be a list.")
            
            if len(variances_pair) != 2:
                raise ValueError(f"{variances_pair} is not a valid input for variances, it needs to be a list of only two elements: [node_number, variance]")
                continue
            
            if variances_pair[0] + 1 > no_nodes:
                raise ValueError(f"{variances_pair[0]} is not a valid input, as there aren't as many nodes, {no_nodes}, in the graphs")
            
            if not isinstance(variances_pair[0], int) or variances_pair[0] < 0:
                raise ValueError(f"{variances_pair[0]} needs to be an int, and non-negative.")
            
            if not isinstance(variances_pair[1], (int, float)):
                raise ValueError(f"{variances_pair[1]} needs to be a (int, float).")

            default_variance[variances_pair[0], variances_pair[0]] = variances_pair[1]
    # pdb.set_trace()
    if correlations is not None:
        if not isinstance(correlations, list):
            raise ValueError(f"{correlations} needs to be a list(list).")
        
        # order them
        for elem in correlations:
            if elem[0] < elem[1]:
                aux_elem = elem[1]
                elem[1] = elem[0]
                elem[0] = aux_elem
        
        for corr in correlations:
            
            if not isinstance(corr, list):
                raise ValueError(f"{corr} needs to be a list.")

            if len(corr) != 3:
                raise ValueError(f"{corr} needs to have length of 3: [distr_1, distr_2, corr_1_2]")
            
            if not isinstance(corr[0], int):
                raise ValueError(f"{corr[0]} needs to be int.")
            
            if not isinstance(corr[1], int):
                raise ValueError(f"{corr[1]} needs to be int.")
            
            if not isinstance(corr[2], (int, float)):
                raise ValueError(f"{corr[2]} needs to be (int, float).")
            
            if corr[2] < 0  or corr[2] > 1:
                raise ValueError(f"{corr[2]} needs to be in interval [0, 1], as it is correlation coeff.")
            # pdb.set_trace()
            default_variance[corr[0], corr[1]] = corr[2] * default_variance[corr[0], corr[0]] * default_variance[corr[1], corr[1]]
            default_variance[corr[1], corr[0]] = corr[2] * default_variance[corr[0], corr[0]] * default_variance[corr[1], corr[1]]

    print(default_variance)
    # if corr[2] > 0.7:
    #     pdb.set_trace()
    try:
        L_cholesky = torch.linalg.cholesky(default_variance)
    except:
        L_cholesky = torch.linalg.cholesky(torch.eye(no_nodes))
        return None
    print(f"Cholesky: {L_cholesky}")
    
    return distr_dict[base_distribution_name.lower()](
            torch.zeros(no_nodes),
            scale_tril=L_cholesky # don't use the covariance_matrix prop, use the scale_tril one, much better!
        )
    

def create_uniform_distr(
        base_distribution_name: str = 'Uniform', 
        uniform_a: list = None,
        uniform_b: list = None,
        no_nodes: int = 5,
        **kwargs
) -> torch.distributions:
    if base_distribution_name.lower() != 'uniform':
        raise NameError()
    

    if uniform_a is not None and uniform_b is not None:
        if len(uniform_a) != len(uniform_b):
            raise ValueError("Please enter same number of values for both a and b.")
        if len(uniform_a) != no_nodes:
            raise ValueError("Please enter values for all nodes.")
        distr = list()
        for a, b in zip(uniform_a, uniform_b):
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise ValueError("Please enter valid values for a and b, need to be int or float.")
            distr.append(distr_dict[base_distribution_name.lower()](
                low=torch.tensor([a], dtype=torch.float),
                high=torch.tensor([b], dtype=torch.float)
            ))
        # pdb.set_trace()
        return Independent(Heterogeneous(distr_list=distr), 1)
    else:
        raise ValueError("Please enter values for a and b, from U(a,b).")


def create_bernoulli_distr(
        base_distribution_name: str = 'Bernoulli',
        bernoulli_coef: float = 0.5,
        **kwargs
) -> torch.distributions:
    # pdb.set_trace()
    x=0


def create_custom_distr(
        base_distribution_name: str = 'Custom',
        multiple_distributions: list = None,
        no_nodes: int = 5,
        **kwargs
) -> torch.distributions:
    if base_distribution_name.lower() != 'custom':
        raise NameError()
    
    if len(multiple_distributions) != no_nodes:
        raise ValueError("Please enter a distribution for each node.")
    
    pdfs = list()

    for pdf in multiple_distributions:
        
        if not isinstance(pdf, list):
            raise ValueError(f"Enter {pdf}, as a list ['Distr name', *params].")
        
        if not isinstance(pdf[0], str):
            raise ValueError(f"First element of {pdf} needs to be a name:str of the pdf.")
        
        if pdf[0].lower() not in [x.lower() for x in acceptable_distr]:
            raise NameError()
        
        if pdf[0].lower() == 'bernoulli'  and len(pdf) != 2:
            raise ValueError("For Bernoulli distribution, you need to enter 2 values, eg. ['Bernoulli', 0.5].")
        elif pdf[0].lower() != 'bernoulli' and len(pdf) != 3:
            raise ValueError(f"For {pdf[0]} you need to enter 3 values, eg. [{pdf[0]}, param1, param2].")
        
        if pdf[0].lower() == 'bernoulli':
            if not isinstance(pdf[1], (int, float)) or pdf[1] < 0 or pdf[1] > 1:
                raise ValueError("Enter valid value for Bernoulli coefficient.")
        else:
            if not isinstance(pdf[1], (int, float)) or not isinstance(pdf[2], (int, float)):
                raise ValueError("Enter valid numbers for distributions.")
            
        if (pdf[0].lower() == 'normal' or pdf[0].lower() == 'laplace') and pdf[2] < 0:
            raise ValueError(f"Enter valid value for {pdf[2]}, non-negative, for {pdf[0]}.")
        
        if pdf[0].lower() == 'bernoulli':
            pdfs.append(distr_dict[pdf[0].lower()](torch.tensor([float(pdf[1])], dtype=torch.float)))
        else:
            pdfs.append(
                distr_dict[pdf[0].lower()](
                    pdf[1],
                    pdf[2]
                )
            )
    # pdb.set_trace()
    return Independent(Heterogeneous(distr_list=pdfs), 1)

            


def list_to_distr(
        base_distribution_name: str = 'Normal', 
        correlations: list = None, 
        means: list = None, 
        variances: list = None, 
        no_nodes: int = 5,
        **kwargs
) -> torch.distributions:
    """
        base_distribution_name: str -> distributions available to make the covariance matrix
        correlations: list(list(float)) -> list of lists, where we have ordered i, j, value
        no_nodes: int -> number of nodes in the SCM
    """
    if base_distribution_name.lower() not in [x.lower() for x in acceptable_distr]:
        raise NameError()
    
    """
        Create a tensor of the size of the largest values in the corresponding i's and j's in the correlations list
        Fill this tensor with the values in correlations
        I.e. [[x, y, val], ...] -> tensor[x, y] == val

        Do Not use covariance_matrix in MultivariateNormal, as it doesn't work as you think it does, i.e. it'll end up
        returning uncorrelated values. Use scale_tril, as apparently it's the one which is internally used anyway by torch
        Scale_tril is lower-triangular factor of covariance, with positive-valued diagonal.        
    """
    # This will crash is no correlations are provided!!!!
    cov_matrix = torch.eye(no_nodes)
    try:
        if not correlations is None:
            for elem in correlations:
                if elem[0] < elem[1]:
                    aux_elem = elem[1]
                    elem[1] = elem[0]
                    elem[0] = aux_elem


            """ Turn the cov_matrix to the L - lower diagonal matrix from Cholesky decomposition
                cov_matrix = L @ L.mT ( due to rounding, it'll be a small difference from the original one
                were we to try L = torch.linalg.cholesky(cov_matrix) and then go back )
            """
    except:
        pass
    # pdb.set_trace()

    if base_distribution_name.lower() == 'multivariatenormal':
        return create_multivariate_normal_dist(base_distribution_name, means, variances, correlations, no_nodes)
    if base_distribution_name.lower() == 'uniform':
        return create_uniform_distr(
            base_distribution_name=base_distribution_name, 
            means=means, 
            variances=variances, 
            correlations=correlations, 
            no_nodes=no_nodes, 
            **kwargs
        )
    if base_distribution_name.lower() == 'bernoulli':
        return create_bernoulli_distr(
            base_distribution_name=base_distribution_name, 
            means=means, 
            variances=variances, 
            correlations=correlations, 
            no_nodes=no_nodes, 
            **kwargs
        )

    if base_distribution_name.lower() == 'custom':
        return create_custom_distr(
            base_distribution_name=base_distribution_name, 
            means=means, 
            variances=variances, 
            correlations=correlations, 
            no_nodes=no_nodes, 
            **kwargs
        )

    else:
        raise ValueError(f"Please enter a valid distribution name. Accepted distributions include {acceptable_distr}")
    return distr_dict[base_distribution_name.lower()](
            torch.zeros(no_nodes),
            scale_tril=cov_matrix # don't use the covariance_matrix prop, use the scale_tril one, much better!
        )


# def base_distribution_3_nodes(name, **kwargs):

def base_distribution_3_nodes(base_distribution_name, correlations, means, variances, no_nodes, base_version=0, **kwargs):
    # if name == "normal_corr":
    #     p_u = normal_corr(**kwargs)
    if base_version == 1:
        p_u = Independent(
            Normal(
                torch.zeros(3),
                torch.ones(3),
            ),
            1,
        )
    elif base_version == 2:
        p_u = Independent(
            Laplace(
                torch.zeros(3),
                torch.ones(3),
            ),
            1,
        )
    elif base_version == 3:
        p_u = Independent(
            Uniform(
                torch.zeros(3),
                torch.ones(3),
            ),
            1,
        )
    elif base_version == 4:
        p_u1 = distr.Normal(loc=torch.tensor([0.0]), scale=1.0)

        mix_u2 = distr.Categorical(torch.ones(1, 2))
        comp_u2 = distr.Normal(
            loc=torch.tensor([[0.0, 1.0]]), scale=torch.tensor([0.2])
        )
        p_u2 = distr.MixtureSameFamily(
            mixture_distribution=mix_u2, component_distribution=comp_u2
        )

        p_u3 = distr.Uniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        p_u = Independent(Heterogeneous(distr_list=[p_u1, p_u2, p_u3]), 1)

    elif base_version == 5: # cov added
        p_u = Independent(
            Laplace(
                torch.zeros(3),
                torch.rand(3),
            ),
            1,
        )

    elif base_version == 6:
        p_u1 = distr.Normal(loc=torch.tensor([0.0]), scale=1.0)

        mix_u2 = distr.Categorical(torch.ones(1, 2))
        comp_u2 = distr.Normal(
            loc=torch.tensor([[0.0, 1.0]]), scale=torch.tensor([0.2])
        )
        p_u2 = distr.MixtureSameFamily(
            mixture_distribution=mix_u2, component_distribution=comp_u2
        )

        p_u3 = distr.Uniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        p_u = Independent(Heterogeneous(distr_list=[p_u1, p_u2, p_u3]), 1)

    elif base_version == 0:
        p_u = list_to_distr(base_distribution_name, correlations, means, variances, no_nodes=no_nodes, **kwargs)
    else:
        raise NotImplementedError(f"Version {base_version} of p_u not implemented.")
    return p_u


def base_distribution_4_nodes(base_distribution_name, correlations, means, variances, no_nodes, base_version=0, **kwargs):
    if base_version == 1:
        p_u = Independent(
            Normal(
                torch.zeros(4),
                torch.ones(4),
            ),
            1,
        )
    elif base_version == 2:
        p_u = Independent(
            Laplace(
                torch.zeros(4),
                torch.ones(4),
            ),
            1,
        )
    elif base_version == 3:
        p_u = Independent(
            Uniform(
                torch.zeros(4),
                torch.ones(4),
            ),
            1,
        )
    elif base_version == 0:
        p_u = list_to_distr(base_distribution_name, correlations, means, variances, no_nodes=no_nodes, **kwargs)

    return p_u


def base_distribution_5_nodes(base_distribution_name, correlations, means, variances, no_nodes, base_version=0, **kwargs):
    if base_version == 1:
        p_u = Independent(
            Normal(
                torch.zeros(5),
                torch.ones(5),
            ),
            1,
        )
    elif base_version == 2:
        p_u = Independent(
            Laplace(
                torch.zeros(5),
                torch.ones(5),
            ),
            1,
        )
    elif base_version == 3:
        p_u = Independent(
            Uniform(
                torch.zeros(5),
                torch.ones(5),
            ),
            1,
        )
    elif base_version == 4:
        p_u = distr.multivariate_normal.MultivariateNormal(
            torch.tensor(
                [0.3, 0.8, 0.9, 1.05, 1.2]
            ),
            torch.tensor([
                [1.4, 0, 0.2, 0., 0.5],
                [0., 1.1, 0., 0., 0.2],
                [0., 0., 1.1, 0., 0.],
                [0., 0., 0., 1.2, 0.],
                [0., 0.4, 0., 0., 1.5]
            ])
        )

    elif base_version == 0:
        
        p_u = list_to_distr(base_distribution_name, correlations, means, variances, no_nodes=no_nodes, **kwargs)
        # p_u = Independent(
        #     Normal(
        #         torch.zeros(5),
        #         torch.ones(5),
        #     ),
        #     1,
        # )  
        
        

    else:
        p_u = Independent(
            Normal(
                torch.zeros(5),
                torch.ones(5),
            ),
            1,
        )  
        

    return p_u


def base_distribution_9_nodes(base_distribution_name, correlations, means, variances, no_nodes, base_version=0, **kwargs):
    if base_version == 1:
        p_u = Independent(
            Uniform(
                1e-6,
                torch.ones(9),
            ),
            1,
        )

    elif base_version == 0:
        p_u = list_to_distr(base_distribution_name, correlations, means, variances, no_nodes=no_nodes, **kwargs)

    elif base_version == 2:
        raise NotImplementedError(f"Version {base_version} of p_u not implemented.")

    return p_u


pu_dict[3] = base_distribution_3_nodes
pu_dict[4] = base_distribution_4_nodes
pu_dict[5] = base_distribution_5_nodes
pu_dict[9] = base_distribution_9_nodes
