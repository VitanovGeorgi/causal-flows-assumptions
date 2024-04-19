import os
import argparse
import pdb
from pathlib import Path
import yaml
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
import ast



parser = argparse.ArgumentParser()
parser.add_argument("--source", help="file path to experiments", default="output_causal_nf/aggregated_results/results__2024-04-16-08:08:25__bcad687afbb711ee8bab2ec714f19cc3")
parser.add_argument("--target", help="output file path", default="output")
parser.add_argument("--metric", help="output file path", default="mmd_obs")
# keep in mind it'll be read as the default is, it's not there because we think it's better this way!
parser.add_argument("--affected_var", help="which nodes are affected by correlation", default=["1", "2"], nargs="+") 

def read_yaml(pth):
    with open(pth, encoding="utf-8") as f:
        return yaml.safe_load(f)

def path_difference(outer_path:str, inner_path:str):
    outer = Path(outer_path)
    inner = Path(inner_path)

    return outer.relative_to(inner)

def unflatten_column_names(df: pd.DataFrame, sym: str = '__') -> pd.DataFrame:
    for name in df.columns:
        if sym in name:
            unflattened_name = name.split(sym)[-1]
            df.rename({name: unflattened_name}, axis=1, inplace=True)
    return df


def cross_reference_variables_df(variables: list, df: pd.DataFrame) -> Tuple[list, pd.DataFrame]:
    
    assert len(variables) > 0, 'No variables were inserted'
    
    _variables = []
    for var in variables:
        try:
            _var = int(var)
        except:
            raise TypeError(f"{var} is not the correct type, it needs to be an integer")
        _variables.append(_var)
    
    _variables = list(set(_variables)) # removing duplicates
    assert len(_variables) > 1, 'Please enter more than 1 different variables'

    _df = df.loc[df['node_count'] >= max(_variables)]

    return _variables, _df


def separate_correlation_into_columns(row: pd.Series) -> pd.Series:
    list_of_lists = ast.literal_eval(row)
    list_of_floats = []
    
    for sublist in list_of_lists:
        list_of_floats.append([float(item) for item in sublist])
    
    correlations = []
    strengths = []
    for sublist in list_of_floats:
        correlations.append([int(sublist[0]), int(sublist[1])])
        strengths.append(sublist[2])

    return pd.Series([correlations, strengths])


if __name__ == "__main__":

    args = parser.parse_args()

    cwd = os.getcwd()
    source_path = os.path.join(cwd, args.source)
    metric = args.metric

    merged_df = pd.read_csv(source_path)
    # keep in mind that the way this file was saved, dataset__{} is still there, etc, so these will be removed
    merged_df = unflatten_column_names(merged_df)

    merged_df['node_count'] = 3 # DELETE!

    vars = args.affected_var

    variables, df = cross_reference_variables_df(vars, merged_df)

    df[['c', 'p']] = df['correlations'].apply(separate_correlation_into_columns)

    # pdb.set_trace()
    split = 'val'
    variable = 'mmd_int_x1=50p'

    """
        What are we doing the avg over?
    """
    elem1 = merged_df.loc[(merged_df['split'] == split) & (merged_df['correlations'] == 0.0)][variable]
    elem2 = merged_df.loc[(merged_df['split'] == split) & (merged_df['correlations'] == 0.3333)][variable]
    elem3 = merged_df.loc[(merged_df['split'] == split) & (merged_df['correlations'] == 0.6667)][variable]
    elem4 = merged_df.loc[(merged_df['split'] == split) & (merged_df['correlations'] == 1.0)][variable]

    elem1 = elem1.sum() / len(elem1)
    elem2 = elem2.sum() / len(elem2)
    elem3 = elem3.sum() / len(elem3)
    elem4 = elem4.sum() / len(elem4)

    """
        'Unnamed: 0', 'current_epoch', 'epoch', 'kl_distance', 'log_prob',
       'log_prob_p', 'log_prob_true', 'loss', 'mmd_int_x1=25p',
       'mmd_int_x1=50p', 'mmd_int_x1=75p', 'mmd_int_x2=25p', 'mmd_int_x2=50p',
       'mmd_int_x2=75p', 'mmd_obs', 'mse_cf_x1=25p', 'mse_cf_x1=50p',
       'mse_cf_x1=75p', 'mse_cf_x2=25p', 'mse_cf_x2=50p', 'mse_cf_x2=75p',
       'rmse_ate_x1=25_50', 'rmse_ate_x1=25_75', 'rmse_ate_x1=50_75',
       'rmse_ate_x2=25_50', 'rmse_ate_x2=25_75', 'rmse_ate_x2=50_75',
       'rmse_cf_x1=25p', 'rmse_cf_x1=50p', 'rmse_cf_x1=75p', 'rmse_cf_x2=25p',
       'rmse_cf_x2=50p', 'rmse_cf_x2=75p', 'time_ate', 'time_cf',
       'time_intervene', 'time_log_prob', 'time_sample_obs', 'timestamp',
       'split', 'base_distribution_name', 'base_version', 'correlations',
       'name', 'num_samples', 'sem_name', 'steps', 'seed'],
      dtype='object')
    """
    
    plt.plot(
        np.linspace(0, 1, 4), 
        [elem1, elem2, elem3, elem4]
    )
    plt.savefig('plot_correlation.png')
    x = 0



    """
        log_prob, log_prob_true, mmd's, rmse_cf, kl_distance - observational distributions
    """




