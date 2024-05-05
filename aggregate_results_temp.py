import os
import argparse
import pdb
from pathlib import Path
import yaml
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import itertools

from typing import Tuple
import ast

import seaborn



parser = argparse.ArgumentParser()
parser.add_argument(
    "--source", help="file path to experiments", 
    default="output_causal_nf/aggregated_results/results_chain"
)
parser.add_argument("--target", help="output file path", default="output_aggregated_results")
parser.add_argument("--metric", help="output file path", default="log_prob")
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
    '''Validates that the inputted df and the variables selected, correspond to each other.

    '''
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

def separate_correlation_into_columns(row: pd.Series, single_strength: bool = True) -> pd.Series:
    list_of_lists = ast.literal_eval(row)
    list_of_floats = []
    
    for sublist in list_of_lists:
        list_of_floats.append([float(item) for item in sublist])
    
    correlations = []
    strengths = []
    for sublist in list_of_floats:
        correlations.append([int(sublist[0]), int(sublist[1])])
        if single_strength:
            strengths = sublist[2]
        else:
            strengths.append(sublist[2])
            
    return pd.Series([correlations, strengths])

def filter_only_selected_vars(row: pd.Series, variables: list) -> pd.Series:
    '''Filters the row to only include the variables that are selected
    '''
    if set(row['c'][0]) == set(variables):
        return row
    

if __name__ == "__main__":

    args = parser.parse_args()

    cwd = os.getcwd()
    source_path = os.path.join(cwd, args.source)
    target_path = os.path.join(cwd, args.target )
    metric = args.metric

    merged_df = pd.read_csv(source_path)
    # keep in mind that the way this file was saved, dataset__{} is still there, etc, so these will be removed
    merged_df = unflatten_column_names(merged_df)

    vars = args.affected_var

    variables, df = cross_reference_variables_df(vars, merged_df)

    df[['c', 'p']] = df['correlations'].apply(separate_correlation_into_columns)

    df_filtered = df.loc[df['c'].apply(lambda x: set(x[0]) == set(variables)) & (df['base_version'] == 0)]

    # pdb.set_trace()
    split = 'test'
    variable = 'mmd_int_x1=50p'
    strength_values = df_filtered['p'].unique()

    # pdb.set_trace()
    assert metric in df_filtered.columns, f"{metric} is not in the columns of the dataframe"


    """
        What are we doing the avg over?
    """


    # elem1 = merged_df.loc[(merged_df['split'] == split) & (merged_df['correlations'] == 0.0)][variable]
    # elem2 = merged_df.loc[(merged_df['split'] == split) & (merged_df['correlations'] == 0.3333)][variable]
    # elem3 = merged_df.loc[(merged_df['split'] == split) & (merged_df['correlations'] == 0.6667)][variable]
    # elem4 = merged_df.loc[(merged_df['split'] == split) & (merged_df['correlations'] == 1.0)][variable]

    # elem1 = elem1.sum() / len(elem1)
    # elem2 = elem2.sum() / len(elem2)
    # elem3 = elem3.sum() / len(elem3)
    # elem4 = elem4.sum() / len(elem4)

    """
        array(['Unnamed: 0', 'current_epoch', 'epoch', 'kl_distance', 'log_prob',
       'log_prob_p', 'log_prob_true', 'loss', 'mmd_int_x1=25p',
       'mmd_int_x1=50p', 'mmd_int_x1=75p', 'mmd_int_x2=25p',
       'mmd_int_x2=50p', 'mmd_int_x2=75p', 'mmd_int_x3=25p',
       'mmd_int_x3=50p', 'mmd_int_x3=75p', 'mmd_obs', 'mse_cf_x1=25p',
       'mse_cf_x1=50p', 'mse_cf_x1=75p', 'mse_cf_x2=25p', 'mse_cf_x2=50p',
       'mse_cf_x2=75p', 'mse_cf_x3=25p', 'mse_cf_x3=50p', 'mse_cf_x3=75p',
       'rmse_ate_x1=25_50', 'rmse_ate_x1=25_75', 'rmse_ate_x1=50_75',
       'rmse_ate_x2=25_50', 'rmse_ate_x2=25_75', 'rmse_ate_x2=50_75',
       'rmse_ate_x3=25_50', 'rmse_ate_x3=25_75', 'rmse_ate_x3=50_75',
       'rmse_cf_x1=25p', 'rmse_cf_x1=50p', 'rmse_cf_x1=75p',
       'rmse_cf_x2=25p', 'rmse_cf_x2=50p', 'rmse_cf_x2=75p',
       'rmse_cf_x3=25p', 'rmse_cf_x3=50p', 'rmse_cf_x3=75p', 'time_ate',
       'time_cf', 'time_intervene', 'time_log_prob', 'time_sample_obs',
       'timestamp', 'split', 'add_noise', 'base_distribution_name',
       'base_version', 'bernoulli_coef', 'correlations', 'k_fold',
       'laplace_diversity', 'loss', 'means', 'multiple_distributions',
       'name', 'num_samples', 'output_plot_metrics', 'root', 'scale',
       'sem_name', 'shuffle_train', 'single_split', 'splits', 'steps',
       'type', 'uniform_a', 'uniform_b', 'use_edge_attr', 'variances',
       'device', 'activate', 'min_delta', 'patience', 'verbose',
       'aggregators', 'dim_inner', 'eps', 'heads', 'num_layers',
       'num_layers_post', 'num_layers_pre', 'post_layers', 'pre_layers',
       'scalers', 'stage_type', 'towers', 'train_eps', 'aggregators',
       'dim_inner', 'eps', 'heads', 'num_layers', 'num_layers_post',
       'num_layers_pre', 'post_layers', 'pre_layers', 'scalers',
       'stage_type', 'towers', 'train_eps', 'act', 'adjacency',
       'base_distr', 'base_to_data', 'beta', 'dim_inner', 'distr_u',
       'distr_x', 'dropout', 'has_bn', 'init', 'lambda_', 'latent_dim',
       'layer_name', 'learn_base', 'name', 'net_name', 'num_layers',
       'objective', 'parity', 'plot', 'scale', 'scale_base', 'shift_base',
       'node_count', 'base_lr', 'beta_1', 'beta_2', 'cooldown', 'factor',
       'gamma', 'mode', 'momentum', 'optimizer', 'patience', 'scheduler',
       'step_size', 'weight_decay', 'param_count', 'root_dir', 'seed',
       'auto_lr_find', 'auto_scale_batch_size', 'batch_size',
       'enable_progress_bar', 'inference_mode', 'kl',
       'limit_train_batches', 'limit_val_batches', 'max_epochs',
       'max_time', 'model_checkpoint', 'num_workers', 'profiler',
       'regularize', 'loss_jacobian_u', 'loss_jacobian_x', 'lr',
       'time_forward', 'c', 'p'], dtype=object)
    """
    
    # colors = iter(cm.rainbow(np.linspace(0, 1, len(df_filtered['num_samples'].unique()))))
    markers = itertools.cycle(["o", "x", "s", "D", "^", "v", "<", ">", "p", "P", "*", "h", "H", "+", "X", "d"])
    colors = itertools.cycle(["r", "b", "g", "m", "y", "c", "k"])
    marker_border = itertools.cycle(["r", "b", "g", "m", "y", "c", "k"])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for split in df_filtered['split'].unique():
        marker = next(markers)
        for seed in df_filtered['seed'].unique():
            color = next(colors)
            for num_samples in df_filtered['num_samples'].unique():
                # print(num_samples)
                marker_border_color = next(marker_border)
                ax.scatter(
                    df_filtered.loc[
                        (df_filtered['num_samples'] == num_samples) & 
                        (df_filtered['seed'] == seed) &
                        (df_filtered['split'] == split)
                    ]['p'], 
                    df_filtered.loc[
                        (df_filtered['num_samples'] == num_samples) & 
                        (df_filtered['seed'] == seed) &
                        (df_filtered['split'] == split)
                    ][metric], 
                    color=color,
                    marker=marker,
                    label=f"samples: {num_samples}, seed: {seed}, split: {split}",
                    edgecolors=marker_border_color
                )
        y = []
        for p in sorted(df_filtered['p'].unique()):
            y.append(df_filtered.loc[
                # (df_filtered['num_samples'] == num_samples) & 
                (df_filtered['p'] == p) & 
                (df_filtered['split'] == split)
            ][metric].mean())
        ax.plot(sorted(df_filtered['p'].unique()), y, color=next(colors), label=f"split: {split}", linestyle='--')
        ax.set_xlabel('Correlation strength')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_ylabel(metric)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    ax.set_title(
        f"SEM: {df_filtered['name'].iloc[0][0]}, metric: {metric}, affected variables: {variables}", 
        backgroundcolor='black', 
        color='white'
        )

    # plt.show()
    ax.legend(prop={'size': 8}, bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=2)
    figure_name = os.path.join(target_path, f"{df_filtered['name'].iloc[0][0]}_metric_{metric}_affected_var_{variables[0]}_{variables[1]}.png")
    fig.savefig(figure_name, dpi=499)
    # plt.show()
    x = 0



    """
        log_prob, log_prob_true, mmd's, rmse_cf, kl_distance - observational distributions
    """




