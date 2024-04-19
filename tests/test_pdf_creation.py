import os

import pytest
import warnings


import torch

from causal_nf.preparators.scm._base_distributions import create_multivariate_normal_dist







@pytest.mark.parametrize("base_distribution_name", [
    'MultivariateNormal', 
    # 'multivariatenormal',
    'MuLtiVariaTENormal',
    # 'MMMMMM'
])
@pytest.mark.parametrize("means", [
    [[0, 1.], [3, 1.], [2, 1.]],
    # [[0, 1.], [3, 1.], [5, 1.]],
    # [[0, 1.], [3, 1.], [2, 1.], [1, 1.2]],
    # [[0, 1.], [3, 1.], [2, 1.], [2, 1.], [2, 1.]]
])
@pytest.mark.parametrize("variances", [
    [[0, 1.], [1, 2.], [1, 3]],
    # [[0, 1.], [1, 2.], [4, 3.]],
    # [[0, 1.], [1, 2.], [1.2, 3]],  
    # [[0, 1.], [1, 2.]]
])
@pytest.mark.parametrize("correlations", [
    # [[0, 1, 1.], [1, 3, 1.], [1, 2, 1.]],
    # [[0, 1, 1.], [1, 3, 1.], [1, 2.2, 1.]],
    # [[0, 1, 1.], [1, 3, 1.], [1, 1, 1.2]],
    # [[2, 1, 1.], [1, 3, 1.], [1, 7, 1.]],
    [[2, 3, 0.5]],
    [[2, 3, 0.]]
])
@pytest.mark.parametrize("no_nodes", [4])
def test_multivariatenormal_pdf_creation(
    base_distribution_name,
    means,
    variances,
    correlations,
    no_nodes
):
    

    # print(f"base: {base_distribution_name}")
    # print(f"means: {means}")
    # print(f"variances: {variances}")
    query_pdf = create_multivariate_normal_dist(base_distribution_name, means, variances, correlations, no_nodes)

    # print(f"hehehe: {query_pdf}")

    # with pytest.raises(Exception) as e_info:
    #     assert issubclass(type(query_pdf), torch.distributions.distribution.Distribution)
    
    if no_nodes == 3:
        pytest.fail("Failure")
    else:
        assert issubclass(type(query_pdf), torch.distributions.distribution.Distribution)
    # assert type(query_pdf) is torch.distributions.distribution.Distribution, f"{query_pdf} is not a torch distribution."

    # assert query_pdf.loc == means[2]


# def test_warning():
#     with pytest.warns(DeprecationWarning):
#         with pytest.warns(UserWarning):  # catches UserWarning, but also every other type too
#             warnings.warn("my warning", UserWarning)
#             warnings.warn("some deprecation warning", DeprecationWarning)
    
    
















