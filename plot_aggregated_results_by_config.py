import os
import argparse
import pdb
from pathlib import Path
import yaml
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import itertools

from typing import Tuple
import ast

import seaborn as sns



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


def node_pairs(_s):
    try:
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
    default="output_causal_nf/aggregated_results/results_chain-2_cov_u_2024-11-12"
)
parser.add_argument("--target", help="output file path", default="output_aggregated_results/chain-2-covariance-rq1")
parser.add_argument("--metric", help="output file path", default="mmd_obs")
# keep in mind it'll be read as the default is, it's not there because we think it's better this way! Also no validation that the nodes are there, or non-negative!
# parser.add_argument("--affected_var", help="which nodes are affected by correlation, \
#                     if none is given then all combinations of nodes are assumed", default=[[(2, 4), (2, 4)], (1, 2)], type=node_pairs, nargs="+") 
# parser.add_argument("--affected_var", help="which nodes are affected by correlation, \
#                     if none is given then all combinations of nodes are assumed", default=[(2, 5)], type=node_pairs, nargs="+") 
parser.add_argument("--affected_var", help="which nodes are affected by correlation, \
                    if none is given then all combinations of nodes are assumed", default='', type=node_pairs, nargs="+") 
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

    merged_df = pd.read_csv(source_path)
    # ['gnn2__dim_inner', 'gnn__dim_inner', 'model__dim_inner'] so to distinguish them, we'll rename them
    merged_df = merged_df.rename(columns={
        'model__dim_inner' : 'model__dim_inner_model',
        'model__num_layers' : 'model__num_layers_model',
    })
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
    # commented out since it'll crush on log_prob_diff == log_prob_true - log_prob
    # assert any(metric in col for col in df.columns), f"{metric} is not in the columns of the dataframe"

    # if only partial matching, then we need to calculate the mean of them
    if metric not in df.columns:
        # df_filtered[metric] = df_filtered.filter(contains=metric).mean(axis=1)
        if 'mse' in metric and 'rmse' not in metric:
            df[metric] = df.filter(like=f"{metric}=").loc[
                :, ~df.filter(like=f"{metric}=").columns.str.contains('rmse')
            ].mean(axis=1) # mean of all mse's since there's 4 of them originally
        elif 'log_prob_diff' in metric:
            df[metric] = df['log_prob_true'] - df['log_prob']
        else:
            df[metric] = df.filter(like=f"{metric}=").mean(axis=1)

    # keep in mind that we are not naming the new columns, so they will be 0, ..., node_count - 1
    # and the columns have nothing to do with where the correlation is, just a way to number them
    df = pd.concat([df, df['correlations'].apply(separate_correlation_into_columns).apply(pd.Series)], axis=1)
    for i in range(1, int(df['node_count'].unique()[0])):
        # in case we don't have multiple correlations at once, for total of all variables
        try:
            df[[f'c{i}', f'p{i}']] = df[i - 1].apply(lambda x: pd.Series([x[0], x[1]]) if isinstance(x, Tuple) else pd.Series([x, x]))
        except:
            df[[f'c{i}', f'p{i}']] = np.nan

    try:
        df['c1'] = df['c1'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    except:
        df['c1'] = np.nan

    try:    
        df['c2'] = df['c2'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    except:
        df['c2'] = np.nan

    try:
        df['c3'] = df['c3'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    except:
        df['c3'] = np.nan
    
    try:
        df['c4'] = df['c4'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    except:
        df['c4'] = np.nan


    """One benefit is, we're still only working with simultaneous change in correlations,
    across all variables, i.e. we don't have one correlation changing, and the other staying the same.
    """

    
    try:
        df_filtered_0 = df.loc[
            df['c1'].notnull() &
            df['c2'].isna() &
            df['c3'].isna() &
            df['c4'].isna()
        ]
        df_filtered_1 = df.loc[
            df['c1'].notnull() &
            df['c2'].isna() &
            df['c3'].isna() &
            df['c4'].isna()
        ]
        df_filtered_2 = df.loc[
            df['c1'].notnull() &
            df['c2'].notnull() &
            df['c3'].isna() &
            df['c4'].isna()
        ]
        df_filtered_3 = df.loc[
            df['c1'].notnull() &
            df['c2'].notnull() &
            df['c3'].notnull() &
            df['c4'].isna()
        ]
        df_filtered_4 = df.loc[
            df['c1'].notnull() &
            df['c2'].notnull() &
            df['c3'].notnull() &
            df['c4'].notnull()
        ]
        df_filtered_5 = df.loc[df['c5'].notnull()]
    except:
        pass
    
    metric_multiples = []
    

    # creating the individual correlations
    # individual_correlations = np.sort(df['c1'].unique())
    # fig, axes = plt.subplots(nrows=len(individual_correlations) // 2, ncols=2, figsize=(8, 8), sharex=True, sharey=True)
    # it = iter(individual_correlations)
        
    try:
        model_param = 'dim_inner_model'
        df_filtered_model = df_filtered_0.sort_values(model_param)
        df_filtered_model = df_filtered_model.loc[
            (df_filtered_model['split'] == 'val') & \
                (df_filtered_model['p1'] == 0)
        ]
        sns_plot_model = sns.relplot(
            data=df_filtered_model[['num_samples', 'split', 'num_layers_model', 'base_to_data', model_param, metric]], 
            x=model_param, 
            y=metric, 
            # row='c1',
            # col='c1',
            errorbar=('ci', 95),
            kind='line', 
            hue='num_layers_model', 
            style='base_to_data',
            palette=["r", "y", "g", "m", "c", "k", "b"], #_palette,
            # col_wrap=int(df_filtered_1['node_count'].unique()[0]),
            facet_kws={'margin_titles': True},
            err_kws={"alpha": .2}
        )

        sns_plot_model.figure.suptitle(f"SEM: {df_filtered_model['name'].iloc[0][0]}", fontsize=12)
        sns_plot_model.figure.subplots_adjust(top=0.95)
        sns_plot_model.tick_params(labelsize=5)
        sns_plot_model.set_xticklabels(rotation=30)
        sns_plot_model.set_axis_labels("Inner dimmension", metric)    
        sns_plot_model._legend.set_title(f"number of layers")
        run_uuid = str(uuid.uuid1()).replace("-", "")
        figure_name = os.path.join(target_path, f"{df_filtered_model['name'].iloc[0][0]}_{metric}_{1}_{run_uuid}.png")

        # for ax in sns_plot_model.axes.flat:
        #     col_label = ax.get_title().split(' = ')[-1]
        #     col_label = eval(col_label)
        #     new_label = f'corr(X{col_label[0] + 1}, X{col_label[1] + 1})'
        #     ax.set_title(new_label)
        
        sns_plot_model.figure.savefig(figure_name)
    except:
        pass

    
    try:
        df_filtered_1 = df_filtered_1.sort_values('c1')
        sns_plot_1 = sns.relplot(
            data=df_filtered_1[['c1', 'p1', 'num_samples', metric]], 
            x='p1', 
            y=metric, 
            # row='c1',
            col='c1',
            errorbar=('ci', 95),
            kind='line', 
            hue='num_samples', 
            style='c1',
            palette=["r", "y", "g", "m", "c", "k", "b"], #_palette,
            # col_wrap=int(df_filtered_1['node_count'].unique()[0]),
            facet_kws={'margin_titles': True},
            err_kws={"alpha": .2}
        )

        # sns_plot_1.figure.suptitle(f"SEM: {df_filtered_1['name'].iloc[0][0]}", fontsize=12)
        # sns_plot_1.figure.subplots_adjust(top=0.95)
        sns_plot_1.set_axis_labels("Correlation strength", metric)    
        sns_plot_1._legend.set_title(f"{df_filtered_1['name'].iloc[0][0]}")
        run_uuid = str(uuid.uuid1()).replace("-", "")
        figure_name = os.path.join(target_path, f"{df_filtered_1['name'].iloc[0][0]}_{metric}_{1}_{run_uuid}.png")

        for ax in sns_plot_1.axes.flat:
            col_label = ax.get_title().split(' = ')[-1]
            col_label = eval(col_label)
            new_label = f'corr(X{col_label[0] + 1}, X{col_label[1] + 1})'
            ax.set_title(new_label)

        # sns_plot_1.figure.savefig(figure_name)
    except:
        pass
        
    # df_filtered_2.groupby(['c1', 'c2']).size().reset_index().rename(columns={0: 'counts'})

    try:
        sns_plot_2 = sns.relplot(
            data=df_filtered_2[['c1', 'c2', 'p1', 'p2', 'num_samples', metric]],
            x='p1',
            y=metric,
            row='c1',
            col='c2',
            errorbar=('ci', 95),
            kind='line',
            hue='num_samples',
            palette=["r", "y", "g", "m", "c", "k", "b"],
            err_kws={"alpha": .2}
        )

        # sns_plot.figure.savefig('sns.jpg')

        # sns_plot_2.figure.suptitle(f"SEM: {df_filtered_2['name'].iloc[0][0]}", fontsize=12)
        # sns_plot_2.figure.subplots_adjust(top=0.95)
        sns_plot_2._legend.set_title(f"{df_filtered_2['name'].iloc[0][0]}")
        sns_plot_2.set_axis_labels("Correlation strength", metric)    
        run_uuid = str(uuid.uuid1()).replace("-", "")
        figure_name = os.path.join(target_path, f"{df_filtered_2['name'].iloc[0][0]}_{metric}_{2}_{run_uuid}.png")

        # for ax in sns_plot_2.axes.flat:
        #     col_label = ax.get_title().split(' = ')[-1]
        #     col_label = eval(col_label)
        #     new_label = f'corr(X{col_label[0] + 1}, X{col_label[1] + 1})'
        #     ax.set_title(new_label)

        sns_plot_2.figure.savefig(figure_name)
    except:
        pass

    try:
        sns_plot_3 = sns.relplot(
                data=df_filtered_3[['c1', 'c2', 'c3', 'p1', 'p2', 'p3', 'num_samples', metric]],
                x='p1',
                y=metric,
                row='c1',
                col='c2',
                style='c3',
                errorbar=('ci', 95),
                kind='line',
                hue='num_samples',
                palette=["r", "y", "g", "m", "c", "k", "b"],
                err_kws={"alpha": .2}
            )

        # sns_plot_3.figure.suptitle(f"SEM: {df_filtered_3['name'].iloc[0][0]}", fontsize=12)
        # sns_plot_3.figure.subplots_adjust(top=0.95)
        sns_plot_3._legend.set_title(f"{df_filtered_3['name'].iloc[0][0]}")
        sns_plot_3.set_axis_labels("Correlation strength", metric)    
        run_uuid = str(uuid.uuid1()).replace("-", "")
        figure_name = os.path.join(target_path, f"{df_filtered_3['name'].iloc[0][0]}_{metric}_{3}_{run_uuid}.png")

        # for ax in sns_plot_3.axes.flat:
        #     col_label = ax.get_title().split(' = ')[-1]
        #     col_label = eval(col_label)
        #     new_label = f'corr(X{col_label[0] + 1}, X{col_label[1] + 1})'
        #     ax.set_title(new_label)

        sns_plot_3.figure.savefig(figure_name)
    except:
        pass

    try:
        sns_plot_4 = sns.relplot(
            data=df_filtered_4[['c1', 'c2', 'c3', 'c4', 'p1', 'p2', 'p3', 'p4', 'num_samples', metric]],
            x='p1',
            y=metric,
            row='c1',
            col='c2',
            style='c3',
            size='c4',
            errorbar=('ci', 95),
            kind='line',
            hue='num_samples',
            palette=["r", "y", "g", "m", "c", "k", "b"],
            err_kws={"alpha": .2}
        )

        # sns_plot_4.figure.suptitle(f"SEM: {df_filtered_4['name'].iloc[0][0]}", fontsize=12)
        # sns_plot_4.figure.subplots_adjust(top=0.95)
        sns_plot_4._legend.set_title(f"{df_filtered_4['name'].iloc[0][0]}")
        sns_plot_4.set_axis_labels("Correlation strength", metric)    
        run_uuid = str(uuid.uuid1()).replace("-", "")
        figure_name = os.path.join(target_path, f"{df_filtered_4['name'].iloc[0][0]}_{metric}_{4}_{run_uuid}.png")

        # for ax in sns_plot_4.axes.flat:
        #     col_label = ax.get_title().split(' = ')[-1]
        #     col_label = eval(col_label)
        #     new_label = f'corr(X{col_label[0] + 1}, X{col_label[1] + 1})'
        #     ax.set_title(new_label)

        sns_plot_4.figure.savefig(figure_name)
    except:
        pass



    """
        log_prob, log_prob_true, mmd's, rmse_cf, kl_distance - observational distributions
    """







