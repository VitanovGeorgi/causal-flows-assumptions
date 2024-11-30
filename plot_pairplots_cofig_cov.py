import os
import argparse
from pathlib import Path
import yaml
import json
import glob
import copy
import uuid
from datetime import datetime


import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import pandas as pd
import wandb


import causal_nf.config as causal_nf_config
import causal_nf.utils.training as causal_nf_train
import causal_nf.utils.wandb_local as wandb_local
from causal_nf.config import cfg
import causal_nf.utils.io as causal_nf_io
from causal_nf.utils.pairwise.mmd import maximum_mean_discrepancy
from causal_nf.utils.pairwise.kl import compute_kl_div, kl2
from causal_nf.utils.pairwise.jsd import jensen_shannon_divergence
from causal_nf.preparators.scm import SCMPreparator





"""
    Given a model (yaml), we will train it, and compare the Pu observed (from the yaml) with the Pu learnt by the model.
"""

parser = argparse.ArgumentParser()

parser.add_argument(
    "--source", help="file path to experiments", 
    default="output_causal_nf/aggregated_results/results_chain_linear_2024-10-26"
)
parser.add_argument("--target", help="output file path", default="output_rq")
parser.add_argument("--correlation", help="for what coorelation we're plotting", default="0")
parser.add_argument("--save", help="whether to plot and save images", default=False)



def read_yaml(pth):
    with open(pth, encoding="utf-8") as f:
        return yaml.safe_load(f)

def path_difference(outer_path:str, inner_path:str):
    outer = Path(outer_path)
    inner = Path(inner_path)

    return outer.relative_to(inner)

def create_output_folder():
    if args.save_metrics_mode == "enabled":
        cwd = os.getcwd()
        metrics_output = os.path.join(cwd, cfg['dataset']['output_plot_metrics'])
        
        # check if results for this name already exist
        for root, dirs, files in os.walk(metrics_output):
            previous_dirs = [w for w in dirs if cfg['dataset']['name'] in w]
            # make new folder with name_{number of old folders + 1}
            dir_no = len(previous_dirs) + 1
            new_dir_name = f"{cfg['dataset']['name']}_{dir_no}"
            # so we don't override existing dirs, just add +1 on the last dirs in there
            while new_dir_name in dirs:
                dir_no += 1
                new_dir_name = f"{cfg['dataset']['name']}_{dir_no}"
            output_dir = os.path.join(metrics_output, f"{cfg['dataset']['name']}_{dir_no}")
            os.mkdir(output_dir)

            return output_dir
            # just need one iteration of this for
            break
    return None

def save_experiment(changing_data, changing_cfg, output_folder: str = None, sign: str = 'U'):
        if not output_folder is None and args.save_metrics_mode == "enabled":
            plt.rcParams.update(plt.rcParamsDefault)
            sns_plot = sns.pairplot(
                    pd.DataFrame(changing_data)
                )
            sns_plot.map_offdiag(sns.kdeplot, levels=4, color=".1")
            
            sns_plot.figure.subplots_adjust(top=0.95)
            sns_plot.figure.suptitle(f"{changing_cfg['name']} {sign} {changing_cfg['correlations']}")
            
            fig_name = f"{changing_cfg['name']}_{changing_cfg['correlations']}_{changing_cfg['type'][:2]}.png"
            sns_plot.figure.savefig(
                os.path.join(output_folder, fig_name)
            )

def reformat_correlations(correlations: list) -> list:
    return [f"corr(U{corr[0] + 1}, U{corr[1] + 1}) = {corr[2]}" for corr in correlations]

class DoSamples:

    def __init__(self, name, index, values) -> None:
        self.name = name
        self.index = index
        self.values = values

def get_do_samples(preparator, model, type: str, x:torch.Tensor) -> tuple:
    
    if type == "interventions" or type == "counterfactual":
        intervention_list = preparator.get_intervention_list()
    elif type == "ate":
        intervention_list = preparator.get_ate_list()
        
    
    x_int_list = []
    x_int_true_list = []

    try:
        for int_dict in intervention_list:
            name = int_dict["name"]
            index = int_dict["index"]
            
            # samples from intervened dist
            if type == "interventions":
                value = int_dict["value"]
                x_int = model.intervene(
                    index=index,
                    value=value,
                    shape=(x.shape[0],),
                    scaler=preparator.scaler_transform,
                )
            elif type == "counterfactual":
                value = int_dict["value"]
                x_int = model.compute_counterfactual(
                    x_factual=x,
                    index=index,
                    value=value,
                    scaler=preparator.scaler_transform,
                )
            elif type == "ate":
                a = int_dict["a"]
                b = int_dict["b"]
                index = int_dict["index"]
                x_int = model.compute_ate(
                    index,
                    a=a,
                    b=b,
                    num_samples=10000,
                    scaler= preparator.scaler_transform,
                )
            else:
                raise ValueError(f"Unsupported type: {type}")
            
            x_int_list.append(DoSamples(name, index, x_int))

            # samples from true dist
            if type == "interventions":
                x_int_true = preparator.intervene(
                    index=index, value=value, shape=(x.shape[0],)
                )
            elif type == "counterfactual":
                x_int_true = preparator.compute_counterfactual(x, index, value)
            elif type == "ate":
                x_int_true = preparator.compute_ate(index, a=a, b=b, num_samples=10000)
            else:
                raise ValueError(f"Unsupported type: {type}")
            
            x_int_true_list.append(DoSamples(name, index, x_int_true))

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    
    return x_int_list, x_int_true_list

def prepare_model(cfg, uncorrelated: bool = False):
    """ Get data from preparator (SEM)
    """
    _cfg = copy.deepcopy(cfg)
    if uncorrelated:
        _cfg.dataset['correlations'] = None
    model_preparator = SCMPreparator.loader(_cfg.dataset)
    X, U = model_preparator.prepare_data(True)
    pdf, corr = model_preparator.get_features_all()

    return model_preparator, X, U, pdf, corr

def train_model(cfg, output_path: str = None):   

    model_preparator, X, U, pdf, corr = prepare_model(cfg)

    # just for now
    model_preparator_0, X_0, U_0, pdf, corr = prepare_model(cfg, True)
    
    model_loaders = model_preparator.get_dataloaders(
        batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
    )
    model = causal_nf_train.load_model(cfg=cfg, preparator=model_preparator)
    run_uuid = str(uuid.uuid1()).replace("-", "")

    if output_path is not None:
        cfg.root_dir = output_path
    
    dirpath = os.path.join(cfg.root_dir, datetime.now().strftime('%Y-%m-%d-%H:%M:%S'), run_uuid)
    logger_dir = os.path.join(cfg.root_dir, run_uuid)
    
    model_trainer, model_logger = causal_nf_train.load_trainer(
        cfg=cfg,
        dirpath=dirpath,
        logger_dir=logger_dir,
        include_logger=True,
        model_checkpoint=cfg.train.model_checkpoint,
        cfg_early=cfg.early_stopping,
        preparator=model_preparator,
    )
    
    model_trainer.fit(model, train_dataloaders=model_loaders[0], val_dataloaders=model_loaders[1])
    ckpt_name_list = ["last"]
    
    if cfg.early_stopping.activate:
        ckpt_name_list.append("best")
    
    for ckpt_name in ckpt_name_list:
        for i, loader_i in enumerate(model_loaders):
            s_name = model_preparator.split_names[i]
            causal_nf_io.print_info(f"Testing {s_name} split")
            model_preparator.set_current_split(i)
            model.ckpt_name = ckpt_name
            _ = model_trainer.test(ckpt_path=ckpt_name, dataloaders=loader_i)
            metrics_stats = model.metrics_stats
            metrics_stats["current_epoch"] = model_trainer.current_epoch
            wandb_local.log_v2(
                {s_name: metrics_stats, "epoch": ckpt_name},
                root=model_trainer.logger.save_dir,
            )

    X_obs = model.input_scaler.inverse_transform(
        model.model.sample((X.shape[0],))['x_obs'], inplace=False
    )

    U_obs = model.input_scaler.inverse_transform(
        model.model.sample((U.shape[0],))['u_obs'], inplace=False
    )

    return X, U, X_obs, U_obs
    
def run_experiments(cfg, output_folder: str = None):
    run = wandb.init(
        mode=args.wandb_mode,
        group=args.wandb_group,
        project=args.project,
        config=config,
    )
    
    X, U, X_obs, U_obs = train_model(cfg, output_folder)

    print(f"X: {torch.corrcoef(X.T)}, \n U: {torch.corrcoef(U.T)}, \n \
          X_obs: {torch.corrcoef(X_obs.T)}, \n U_obs: {torch.corrcoef(U_obs.T)}")
    i = 0




if __name__ == "__main__":

    os.environ["WANDB_NOTEBOOK_NAME"] = "name_of_the_notebook"

    args_list, args = causal_nf_config.parse_args()

    args.save_metrics_mode = "enabled" # "enabled" or "disabled" comment out

    load_model = isinstance(args.load_model, str)
    if load_model:
        causal_nf_io.print_info(f"Loading model: {args.load_model}")

    config = causal_nf_config.build_config(
        config_file= "grids/causal_nf/comparison_distributions/model_1.yaml", # args.config_file,
        args_list=args_list,
        config_default_file=args.config_default_file,
    )


    causal_nf_config.assert_cfg_and_config(cfg, config)

    if cfg.device in ["cpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    causal_nf_train.set_reproducibility(cfg)




    output_folder = create_output_folder()

    run_experiments(cfg, output_folder)