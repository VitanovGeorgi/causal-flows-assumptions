import os
import argparse
import pdb
from pathlib import Path
import yaml
import json
import uuid
from datetime import datetime


import numpy as np
import pandas as pd

import causal_nf.utils.io as causal_io



parser = argparse.ArgumentParser()
parser.add_argument("--source", help="file path to experiments", default="output_causal_nf/comparison_x_u/fork_non_lin")
parser.add_argument("--target", help="output file path, please don't use the same as the source file, since that dir will be overwritten when the next experiment is run", default="output_causal_nf")
parser.add_argument("--subtarget", help="name of aggregated output dir", default="aggregated_results")
parser.add_argument("--name", help="name of aggregated output csv file", default="results")

def read_yaml(pth):
    with open(pth, encoding="utf-8") as f:
        return yaml.safe_load(f)

def path_difference(outer_path:str, inner_path:str):
    outer = Path(outer_path)
    inner = Path(inner_path)

    return outer.relative_to(inner)

def nested_dict_to_pd(nested_dict: dict) -> pd.DataFrame:
    '''
        We know that the nested_dict has a structure 
       {
            test: {
                ...
            },
            epoch: str,
            timestamps: int
       } 
    '''
    unrolled_dict = dict()
    _type = None
    for i in nested_dict.keys():
        try:
            for j in nested_dict[i].keys():
                unrolled_dict[j] = nested_dict[i][j] 
                _type = i
        except AttributeError:
                unrolled_dict[i] = nested_dict[i] 

    # try:
    output_pd = pd.DataFrame.from_dict(
        dict([ (k,pd.Series(v)) for k,v in unrolled_dict.items() ]), 
        orient='index'
    ).T
    # except:
        # x = 0
    output_pd['split'] = _type
    
    return output_pd



if __name__ == "__main__":

    args = parser.parse_args()

    cwd = os.getcwd()
    source_path = os.path.join(cwd, args.source)
    target_path = os.path.join(cwd, args.target)
    output_dir = os.path.join(target_path, args.subtarget)
    

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    



    # gather the data
    all_experiments_results = dict()
    all_experiments_params = dict()
    all_experiments_logs = dict()
    single_experiment = list()
    for root, dirs, files in os.walk(source_path):
        try:
            if 'metrics.csv' in files:
                single_metric = pd.read_csv(os.path.join(root, 'metrics.csv'))
                difference_in_path = path_difference(root, source_path)
                all_experiments_results[difference_in_path.parts[0]] = single_metric.iloc[-3:]
            if 'config_local.yaml' in files:
                cfg = read_yaml(os.path.join(root, 'config.yaml'))
                difference_in_path = path_difference(root, source_path)
                all_experiments_params[difference_in_path.parts[0]] = cfg
            if 'logs.txt' in files:
                txt_file = open(os.path.join(root, 'logs.txt'))
                content = txt_file.readlines()
                '''
                    We already know that the last three elements will be those three.
                '''
                test_pd = nested_dict_to_pd(json.loads(content[-1]))
                val_pd = nested_dict_to_pd(json.loads(content[-2]))
                train_pd = nested_dict_to_pd(json.loads(content[-3]))
                difference_in_path = path_difference(root, source_path)
                all_experiments_logs[difference_in_path.parts[0]] = pd.concat([train_pd.iloc[0], test_pd.iloc[0], val_pd.iloc[0]], axis=1).T
        except Exception as e:
            continue
    # add the params columns to the results
    for key in all_experiments_logs.keys():
        try:
            params_df = pd.DataFrame.from_dict(all_experiments_params[key], orient='index').T
            all_experiments_logs[key] = pd.concat([all_experiments_logs[key], params_df], axis=1)
        except:
            continue

    # merge the data into a single df
    merged_df = pd.concat([all_experiments_logs[key] for key in all_experiments_logs.keys()], axis=0)

    run_uuid = str(uuid.uuid1()).replace("-", "")
    output_file = '__'
    output_file = output_file.join([args.name, datetime.now().strftime('%Y-%m-%d-%H:%M:%S'), run_uuid])
    # output_file = args.name + '_' + all_experiments_logs['sem_name'] + '_' + all_experiments_logs['base_distribution_name'] + '___' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '___' + run_uuid
    file_name = os.path.join(target_path, output_dir, output_file)
    # pdb.set_trace()
    merged_df.to_csv(file_name, encoding='utf-8')

"""
    log_prob, log_prob_true, mmd's, rmse_cf, kl_distance - observational distributions
"""



