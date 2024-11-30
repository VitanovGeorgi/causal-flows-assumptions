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

import seaborn as sns



parser = argparse.ArgumentParser()
parser.add_argument(
    "--source", help="file path to experiments", 
    default="output_causal_nf/aggregated_results/results_chain_linear_2024-10-26"
)
parser.add_argument("--target", help="output file path", default="output_aggregated_results")
parser.add_argument("--metric", help="output file path", default="kl_distance")
parser.add_argument("--plot", help="whether to plot and save images", default=False)
# keep in mind it'll be read as the default is, it's not there because we think it's better this way!
parser.add_argument("--affected_var", help="which nodes are affected by correlation", default=["1", "0"], nargs="+") 
parser.add_argument("--changed_params", help="which params were changed in the config", default=["base_lr", "dim_inner_model"], nargs="+") 


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
    target_path = os.path.join(cwd, args.target)
    metric = args.metric

    merged_df = pd.read_csv(source_path)
    # ['gnn2__dim_inner', 'gnn__dim_inner', 'model__dim_inner'] so to distinguish them, we'll rename them
    merged_df = merged_df.rename(columns={'model__dim_inner' : 'model__dim_inner_model'})
    # keep in mind that the way this file was saved, dataset__{} is still there, etc, so these will be removed
    merged_df = unflatten_column_names(merged_df)

    vars = args.affected_var
    params = args.changed_params
    _plot_flag = args.plot  

    variables, df = cross_reference_variables_df(vars, merged_df)

    df[['c', 'p']] = df['correlations'].apply(separate_correlation_into_columns)

    df_filtered = df.loc[df['c'].apply(lambda x: set(x[0]) == set(variables)) & (df['base_version'] == 0)]

    # pdb.set_trace()
    split = 'val'
    variable = 'mmd_int_x1=50p'
    strength_values = df_filtered['p'].unique()

    assert metric in df_filtered.columns, f"{metric} is not in the columns of the dataframe"


    # colors = iter(cm.rainbow(np.linspace(0, 1, len(df_filtered['num_samples'].unique()))))
    markers = itertools.cycle(["o", "x", "s", "D", "^", "v", "<", ">", "p", "P", "*", "h", "H", "+", "X", "d"])
    colors = itertools.cycle(["r", "b", "g", "m", "y", "c", "k"])
    marker_border = itertools.cycle(["r", "b", "g", "m", "y", "c", "k"])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    _x_axis_metric = 'dim_inner_model'

    colors_dict = {}
    for dim in df_filtered['dim_inner_model'].unique():
        colors_dict[dim] = next(colors)

    metric_dict = {}
    for dim in df_filtered['dim_inner_model'].unique():
        df_filtered_local = df_filtered.loc[df_filtered['dim_inner_model'] == dim]
        for lr in df_filtered_local['base_lr'].unique():
            df_filtered_local_lr = df_filtered_local.loc[(df_filtered_local['base_lr'] == lr) & (df_filtered_local['split'] == split)]
            color = next(colors)
            sns_plot = sns.regplot(
                data=df_filtered_local_lr,
                x='p',
                y=metric,
                ax=ax,
                # scatter_kws={'s': 10, 'se': 2},
                line_kws={
                    'color': colors_dict[dim], 
                    # 'label': f"dim: {dim}, lr: {lr}"
                },
                scatter=False,
                label=f"dim: {dim}, lr: {lr}" # if scatter is True, this will be the label of the scatter
            )
            ax.set_xlabel('Learning rate')
            ax.set_ylabel(metric)
            for p in df_filtered_local_lr['p'].unique():
                if p in metric_dict:
                    metric_dict[p].append((dim, lr, df_filtered_local_lr.loc[df_filtered_local_lr['p'] == p][metric].mean()))
                else:
                    metric_dict[p] = [(dim, lr, df_filtered_local_lr.loc[df_filtered_local_lr['p'] == p][metric].mean())]
                
        best_metric_at_each_corr = []
        for p in np.sort(df_filtered_local_lr['p'].unique()):
            ''' Should change min or max, depending on the metric
            '''
            best_metric_at_each_corr.append(min(metric_dict[p], key=lambda x: x[2]))
        
    sns_plot.set_title(f"{df_filtered_local['name'].iloc[0][0]}\n {[(elem[0], elem[1]) for elem in best_metric_at_each_corr]}")        
    sns_plot.legend(
        prop={'size': 8}, bbox_to_anchor=(0.5, -0.05), 
        loc='upper center', ncol=2
    )
    figure_name = os.path.join(target_path, f"{df_filtered['name'].iloc[0][0]}_metric_{metric}_affected_var_{variables[0]}_{variables[1]}.png")
    plt.show()
    if _plot_flag:
        fig.savefig(figure_name, dpi=499)
    x = 0





