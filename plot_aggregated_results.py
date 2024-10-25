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
# from tueplots import bundles
# from tueplots import figsizes, fontsizes



def node_pairs(_s):
    try:
        # pdb.set_trace()
        if _s != '': 
            result = map(int, _s.split(','))
            result_list = list(result)
            if len(result_list) == 2:
                x, y = map(int, _s.split(','))
                return x, y
            elif len(result_list) % 2 == 0:
                result = map(int, _s.split(','))
                return list(zip(*[iter(result)]*2))
        else:
            return ''
    except:
        raise argparse.ArgumentTypeError("Node pairs must be x,y")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source", help="file path to experiments", 
    default="output_causal_nf/aggregated_results/results_chain_3_2024-07-11-14"
)
parser.add_argument("--target", help="output file path", default="output_aggregated_results/chain5_incomplete")
parser.add_argument("--metric", help="output file path", default="mse_cf_x1")
# keep in mind it'll be read as the default is, it's not there because we think it's better this way! Also no validation that the nodes are there, or non-negative!
parser.add_argument("--affected_var", help="which nodes are affected by correlation, \
                    if none is given then all combinations of nodes are assumed", default=[[(1, 2), (2, 3)], (2, 3)], type=node_pairs, nargs="+") 
# parser.add_argument("--affected_var", help="which nodes are affected by correlation, \
#                     if none is given then all combinations of nodes are assumed", default=[(1, 5)], type=node_pairs, nargs="+") 
# parser.add_argument("--affected_var", help="which nodes are affected by correlation, \
#                     if none is given then all combinations of nodes are assumed", default='', type=node_pairs, nargs="+") 
parser.add_argument("--split", help="which split to plot", default='test')

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
    
    _variables = []
    for var in variables:
        # when we put them in, we start counting from 1, so we need to subtract 1
        if isinstance(var, tuple):
            if isinstance(var[0], int) and isinstance(var[1], int):
                _variables.append((var[0] - 1, var[1] - 1))
            else:
                raise TypeError(f"{var} is not the correct type, it needs to be a tuple of integers")
        elif isinstance(var, list):
            _sub_variables = []
            for item in var:
                if isinstance(item[0], int) and isinstance(item[1], int):
                    _sub_variables.append((item[0] - 1, item[1] - 1))
                else:
                    raise TypeError(f"{item} is not the correct type, it needs to be a tuple of integers")
            _variables.append(_sub_variables)
        else:
            raise TypeError(f"{var} is not the correct type, it needs to be a tuple, or list of tuples of integers")

    max_list = []
    for item in _variables:
        if isinstance(item, list):
            for sub_item in item:
                max_list.append(max(sub_item))
        elif isinstance(item, tuple):
            max_list.append(max(item))
        else:
            max_list.append(item)

    _df = df.loc[df['node_count'] > max(max_list)]

    return _variables, _df

def separate_correlation_into_columns(row: pd.Series, single_strength: bool = False) -> pd.Series:
    list_of_lists = ast.literal_eval(row)
    list_of_floats = []
    
    # they are strings, so we need to convert them to floats
    for sublist in list_of_lists:
        list_of_floats.append([float(item) for item in sublist])
    
    correlations = []
    strengths = []
    for sublist in list_of_floats:
        
        if single_strength:
            strengths = sublist[2]
            correlations = (sorted(int(sublist[0]), int(sublist[1])))
        else:
            strengths.append(sublist[2])
            correlations.append(sorted([int(sublist[0]), int(sublist[1])]))
            
            
    # return pd.Series([correlations, strengths])
    return pd.Series(list(zip(correlations, strengths)))
    # return [pd.Series([c, p]) for c, p in zip(correlations, strengths)]

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
    split = args.split
    # pdb.set_trace()
    merged_df = pd.read_csv(source_path)
    # keep in mind that the way this file was saved, dataset__{} is still there, etc, so these will be removed
    merged_df = unflatten_column_names(merged_df)

    vars = args.affected_var
    if vars == '':
        vars = list(itertools.combinations(range(1, merged_df['node_count'].unique()[0] + 1), 2))
    
    variables, df = cross_reference_variables_df(vars, merged_df)
    # pdb.set_trace()
    # so we know that the correlations are sorted, we need this assumption for later
    df.sort_values('correlations')

    # partial matching of the metric
    assert any(metric in col for col in df.columns), f"{metric} is not in the columns of the dataframe"

    # if only partial matching, then we need to calculate the mean of them
    if metric not in df.columns:
        # df_filtered[metric] = df_filtered.filter(contains=metric).mean(axis=1)
        if 'mse' in metric and 'rmse' not in metric:
            df[metric] = df.filter(like=f"{metric}=").loc[
                :, ~df.filter(like=f"{metric}=").columns.str.contains('rmse')
            ].mean(axis=1)
        else:
            df[metric] = df.filter(like=f"{metric}=").mean(axis=1)

    # keeo in mind that we are not naming the new columns, so they will be 0, ..., node_count - 1
    df = pd.concat([df, df['correlations'].apply(separate_correlation_into_columns).apply(pd.Series)], axis=1)
    for i in range(1, int(df['node_count'].unique()[0])):
        df[[f'c{i}', f'p{i}']] = df[i - 1].apply(lambda x: pd.Series([x[0], x[1]]) if isinstance(x, Tuple) else pd.Series([x, x]))

    # df[['cp1', 'cp2', 'cp3', 'cp4']] = df['correlations'].apply(separate_correlation_into_columns)
    # df[['c1', 'p1']] = df['cp1'].apply(lambda x: pd.Series([x[0], x[1]]))
    # df[['c2', 'p2']] = df['cp2'].apply(lambda x: pd.Series([x[0], x[1]]))
    # df[['c3', 'p3']] = df['cp3'].apply(lambda x: pd.Series([x[0], x[1]]))
    # df[['c4', 'p4']] = df['cp4'].apply(lambda x: pd.Series([x[0], x[1]]))




    # df['c'] = df['c'].apply(lambda x: tuple(sorted(x)))

    # df_filtered = df.loc[(df['c'].apply(lambda x: set(df['c'][0]) in [set(v) for v in variables])) & (df['base_version'] == 0)]
    # df_filtered = df.loc[(df['c'].apply(lambda x: x in variables)) & (df['base_version'] == 0)]
    try:
        df_filtered_1 = df.loc[df['c1'].notnull()]
        df_filtered_2 = df.loc[df['c2'].notnull()]
        df_filtered_3 = df.loc[df['c3'].notnull()]
        df_filtered_4 = df.loc[df['c4'].notnull()]
        df_filtered_5 = df.loc[df['c5'].notnull()]
    except:
        pass

    metric_multiples = []

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
    
    
    for var_pairs in variables:
        confidence_interval = 95
        # these are tuple pairs, so * 2
        if isinstance(var_pairs, tuple):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            for num_samples in -np.sort(-df_filtered_1['num_samples'][::-1].unique()):
                
                _df = df_filtered_1.loc[
                        (df_filtered_1['num_samples'] == num_samples) & 
                        (df_filtered_1['split'] == split) & 
                        (df_filtered_1['c1'].apply(lambda x: x == list(var_pairs)))
                    ].sort_values('p1')
                _df = _df[(_df['c2'].isna()) & (_df['c3'].isna()) & (_df['c4'].isna())]
                
                if len(_df) == 0:
                    continue

                _xaxis = _df['p1'].unique()
                _yaxis = _df.groupby(['p1'])[metric].mean()
        
            try:
                # we might not have any data for this var_pairs
                data = pd.DataFrame({'correlations': _xaxis, '_yaxis': _yaxis})

                plt.rcParams.update(plt.rcParamsDefault)
                
                _palette = sns.color_palette("Set2")
                df_filtered_1['c1'] = df_filtered_1['c1'].apply(lambda x: tuple(x))
                df_local_filtered = df_filtered_1.loc[
                    (df_filtered_1['c1'].apply(lambda x: x == var_pairs))
                ]
                df_local_filtered = df_local_filtered[
                    (df_local_filtered['c4'].isna()) &
                    (df_local_filtered['c3'].isna()) &
                    (df_local_filtered['c2'].isna())
                ]
                sns_plot = sns.relplot(
                    data=df_local_filtered[['c1', 'p1', 'num_samples', metric]], 
                    x='p1', 
                    y=metric, 
                    errorbar=('ci', confidence_interval),
                    kind='line', 
                    hue='num_samples', 
                    style='c1',
                    palette=_palette, # ["r", "y", "g", "m", "c", "k", "b"], #_palette,
                    err_kws={"alpha": .2}
                )

                sns_plot.figure.suptitle(f"SEM: {df_filtered_1['name'].iloc[0][0]}, ci: {confidence_interval}%, affected vars: {var_pairs}", fontsize=12)
                sns_plot.figure.subplots_adjust(top=0.95)
                sns_plot.set_axis_labels("Correlation strength", metric)    

                figure_name = os.path.join(target_path, f"{df_filtered_1['name'].iloc[0][0]}_metric_{metric}_affected_var_{var_pairs}.png")
                # sns_plot.figure.savefig(figure_name)
            except:
                pass

        elif len(var_pairs) == 2 and isinstance(var_pairs, list):
            df_local_filtered = df_filtered_2.loc[
                (df_filtered_2['c1'].apply(lambda x: x == list(var_pairs[0]))) & 
                (df_filtered_2['c2'].apply(lambda x: x == list(var_pairs[1])))
            ]
            df_local_filtered = df_local_filtered[(df_local_filtered['c3'].isna()) & (df_local_filtered['c4'].isna())]
            for num_samples in -np.sort(-df_filtered_2['num_samples'][::-1].unique()):
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                _df_1 = df_local_filtered.loc[
                        (df_local_filtered['num_samples'] == num_samples) & 
                        (df_local_filtered['split'] == split)
                    ].sort_values('p1')
                _df_2 = df_local_filtered.loc[
                        (df_local_filtered['num_samples'] == num_samples) & 
                        (df_local_filtered['split'] == split)
                    ].sort_values('p2')
                arr = np.zeros((len(_df_1['p1'].unique()), len(_df_2['p2'].unique())))
                off_diagonal_flag = False
                for i, p1 in enumerate(_df_1['p1'].unique()):
                    for j, p2 in enumerate(_df_2['p2'].unique()):
                        try:
                            arr[i, j] = _df_1.loc[(_df_1['p1'] == p1) & (_df_1['p2'] == p2), metric].mean() # mean of seeds
                            if p1 != p2 and not np.isnan(arr[i, j]):
                                off_diagonal_flag = True
                        except:
                            # we don't have all combinations of p1, p2, so if no combination there, assign 0
                            arr[i, j] = None
                
                if off_diagonal_flag:
                    sns_plot = sns.heatmap(
                        arr, 
                        annot=True, 
                        fmt=".2f", 
                        cmap="YlGnBu", 
                        mask=np.isnan(arr),
                        xticklabels=sorted(_df_1['p1'].unique()),
                        yticklabels=sorted(_df_2['p2'].unique()),
                        cbar_kws={'label': metric}
                    )
                    sns_plot.set_title(f"SEM: {_df_1['name'].iloc[0][0]}, affected vars: {var_pairs}, num_samples: {num_samples}")
                    sns_plot.set_xlabel(f"Correlation between {var_pairs[0]}")
                    sns_plot.set_ylabel(f"Correlation between {var_pairs[1]}")

                    figure_name = os.path.join(target_path, f"{_df_1['name'].iloc[0][0]}_metric_{metric}_affected_var_{var_pairs}_num_samples{num_samples}.png")
                    # sns_plot.figure.savefig(figure_name)
            
            if not off_diagonal_flag:
                sns_plot = sns.relplot(
                    data= df_local_filtered[['p1', 'num_samples', metric]],
                    x='p1',
                    y=metric,
                    errorbar=('ci', confidence_interval),
                    kind='line', 
                    hue='num_samples',
                    marker='o',
                    markersize=3,
                    linewidth=1,
                    palette=["r", "y", "g", "m", "c", "k", "b"]
                )

                sns_plot.figure.suptitle(f"SEM: {df_local_filtered['name'].iloc[0][0]}, ci: {confidence_interval}%, affected vars: {var_pairs}", fontsize=12)
                sns_plot.figure.subplots_adjust(top=0.95)
                sns_plot.set_axis_labels("Correlation strength", metric)    


                figure_name = os.path.join(target_path, f"{_df_1['name'].iloc[0][0]}_metric_{metric}_affected_var_{var_pairs}_num_samples{num_samples}.png")
                # sns_plot.figure.savefig(figure_name)

        elif len(var_pairs) == 3:
            df_local_filtered = df_filtered_3.loc[
                (df_filtered_3['c1'].apply(lambda x: x == list(var_pairs[0]))) & 
                (df_filtered_3['c2'].apply(lambda x: x == list(var_pairs[1]))) &
                (df_filtered_3['c3'].apply(lambda x: x == list(var_pairs[2])))
            ]
            df_local_filtered = df_local_filtered[(df_local_filtered['c4'].isna())]
            for num_samples in -np.sort(-df_filtered_3['num_samples'][::-1].unique()):
                # fig = plt.figure(figsize=(8, 8))
                # ax = fig.add_subplot(111)
                _df_1 = df_local_filtered.loc[
                        (df_local_filtered['num_samples'] == num_samples) & 
                        (df_local_filtered['split'] == split)
                    ].sort_values('p1')
                _df_2 = df_local_filtered.loc[
                        (df_local_filtered['num_samples'] == num_samples) & 
                        (df_local_filtered['split'] == split)
                    ].sort_values('p2')
                _df_3 = df_local_filtered.loc[
                        (df_local_filtered['num_samples'] == num_samples) & 
                        (df_local_filtered['split'] == split)
                    ].sort_values('p3')
                
                arr = np.zeros((len(_df_1['p1'].unique()), len(_df_2['p2'].unique()), len(_df_3['p3'].unique())))
                off_diagonal_flag = False
                for i, p1 in enumerate(sorted(_df_1['p1'].unique())):
                    for j, p2 in enumerate(sorted(_df_2['p2'].unique())):
                        for k, p3 in enumerate(sorted(_df_3['p3'].unique())):
                            try:
                                arr[i, j, k] = _df_1.loc[(_df_1['p1'] == p1) & (_df_1['p2'] == p2) & (_df_1['p3'] == p3), metric].mean() # mean of seeds
                                if (p1 != p2 or p1 != p3 or p2 != p3) and not np.isnan(arr[i, j, k]):
                                    off_diagonal_flag = True
                            except:
                                # we don't have all combinations of p1, p2, so if no combination there, assign 0
                                arr[i, j, k] = None
                                
                if off_diagonal_flag:
                    fig, axn = plt.subplots(len(_df_3['p3'].unique()), len(_df_3['p3'].unique()), sharex=True, sharey=True)
                    cbar_ax = fig.add_axes([.91, .3, .03, .4])

                    for i, ax in enumerate(axn.flat):
                        if i // len(_df_3['p3'].unique()) == 0:
                            data = arr[i, :, :]
                            xlabel = 'Correlation ' + str(var_pairs[1])
                            ylabel = 'Correlation ' + str(var_pairs[2])
                            current_title = f"Correlation {var_pairs[0]}: {sorted(_df_1['p1'].unique())[i]}"
                            _xticklabels = sorted(_df_2['p2'].unique())
                            _yticklabels = sorted(_df_3['p3'].unique())
                        elif i // len(_df_3['p3'].unique()) == 1:
                            data = arr[:, i % len(_df_3['p3'].unique()), :]
                            xlabel = 'Correlation ' + str(var_pairs[0])
                            ylabel = 'Correlation ' + str(var_pairs[2])
                            current_title = f"Correlation {var_pairs[1]}: {sorted(_df_2['p2'].unique())[i % len(_df_3['p3'].unique())]}"
                            _xticklabels = sorted(_df_1['p1'].unique())
                            _yticklabels = sorted(_df_3['p3'].unique())
                        else:
                            data = arr[:, :, i % len(_df_3['p3'].unique())]
                            xlabel = 'Correlation ' + str(var_pairs[0])
                            ylabel = 'Correlation ' + str(var_pairs[1])
                            current_title = f"Correlation {var_pairs[2]}: {sorted(_df_3['p3'].unique())[i % len(_df_3['p3'].unique())]}"
                            _xticklabels = sorted(_df_1['p1'].unique())
                            _yticklabels = sorted(_df_2['p2'].unique())
                        sns_plot = sns.heatmap(
                            data, 
                            annot=True,
                            fmt=".2f", 
                            cmap="YlGnBu", 
                            mask=np.isnan(data),
                            ax=ax,
                            cbar= i == 0,
                            cbar_ax=None if i else cbar_ax,
                            cbar_kws={'label': metric},
                            xticklabels=_xticklabels,
                            yticklabels=_yticklabels
                        )
                        ax.set_title(current_title, fontsize=5)
                        ax.set_xlabel(xlabel, fontsize=5)
                        ax.set_ylabel(ylabel, fontsize=5)

                    fig.tight_layout(rect=[0, 0, 1.1, 1.1])

                    sns_plot.figure.suptitle(f"SEM: {df_local_filtered['name'].iloc[0][0]}, ci: {confidence_interval}%, affected vars: {var_pairs}", fontsize=12)
                    sns_plot.figure.subplots_adjust(top=0.95)
                    sns_plot.set_axis_labels("Correlation strength", metric)    

                    figure_name = os.path.join(target_path, f"{_df_1['name'].iloc[0][0]}_metric_{metric}_affected_var_{var_pairs}_num_samples{num_samples}.png")
                    # sns_plot.figure.savefig(figure_name)

            if not off_diagonal_flag:
                sns_plot = sns.relplot(
                    data= df_local_filtered[['p1', 'num_samples', metric]],
                    x='p1',
                    y=metric,
                    errorbar=('ci', confidence_interval),
                    kind='line', 
                    hue='num_samples',
                    marker='o',
                    markersize=3,
                    linewidth=1,
                    palette=["r", "y", "g", "m", "c", "k", "b"]
                )

                sns_plot.figure.suptitle(f"SEM: {df_local_filtered['name'].iloc[0][0]}, ci: {confidence_interval}%, affected vars: {var_pairs}", fontsize=12)
                sns_plot.figure.subplots_adjust(top=0.95)
                sns_plot.set_axis_labels("Correlation strength", metric)    


                figure_name = os.path.join(target_path, f"{_df_1['name'].iloc[0][0]}_metric_{metric}_affected_var_{var_pairs}_num_samples{num_samples}.png")
                # sns_plot.figure.savefig(figure_name)

            
        else:
            pass
        
        

    
    
    
       

    # fig.savefig(figure_name, dpi=499)
    # plt.show()
    x = 0



    """
        log_prob, log_prob_true, mmd's, rmse_cf, kl_distance - observational distributions
    """




