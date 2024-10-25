import glob

import wandb
import os
import causal_nf.config as causal_nf_config
import causal_nf.utils.training as causal_nf_train
import causal_nf.utils.wandb_local as wandb_local
from causal_nf.config import cfg
import causal_nf.utils.io as causal_nf_io
from causal_nf.utils.pairwise.mmd import maximum_mean_discrepancy
from causal_nf.utils.pairwise.kl import compute_kl_div, kl2
from causal_nf.utils.pairwise.jsd import jensen_shannon_divergence

import pdb
import copy
import uuid

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import seaborn as sns


import numpy as np
import pandas as pd


os.environ["WANDB_NOTEBOOK_NAME"] = "name_of_the_notebook"

args_list, args = causal_nf_config.parse_args()

load_model = isinstance(args.load_model, str)
if load_model:
    causal_nf_io.print_info(f"Loading model: {args.load_model}")

config = causal_nf_config.build_config(
    config_file= "grids/causal_nf/comparison_distributions/compare_distributions.yaml", # args.config_file,
    args_list=args_list,
    config_default_file=args.config_default_file,
)

causal_nf_config.assert_cfg_and_config(cfg, config)

if cfg.device in ["cpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
causal_nf_train.set_reproducibility(cfg)

if cfg.dataset.name in ["german"]:
    from causal_nf.preparators.german_preparator import GermanPreparator

    preparator = GermanPreparator.loader(cfg.dataset)
elif cfg.dataset.name in ["ihdp"]:
    from causal_nf.preparators.ihdp_preparator import IHDPPreparator

    preparator = IHDPPreparator.loader(cfg.dataset)
else:
    from causal_nf.preparators.scm import SCMPreparator
    # pdb.set_trace()
    
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



    def run_experiments(cfg, output_folder: str = None):
        run = wandb.init(
            mode=args.wandb_mode,
            group=args.wandb_group,
            project=args.project,
            config=config,
        )

        # # creating uncorrelated data
        # uncorrelated_cfg = copy.deepcopy(cfg)

        # uncorrelated_cfg.dataset['correlations'] = None
        # uncorrelated_preparator = SCMPreparator.loader(uncorrelated_cfg.dataset)
        # X_uncorrelated, U_uncorrelated = uncorrelated_preparator.prepare_data(True)
        # pdf_uncorrelated, corr_uncorrelated = uncorrelated_preparator.get_features_all()
        
        # uncorrelated_loaders = uncorrelated_preparator.get_dataloaders(batch_size=uncorrelated_cfg.train.batch_size, num_workers=uncorrelated_cfg.train.num_workers)
        # uncorrelated_model = causal_nf_train.load_model(cfg=uncorrelated_cfg, preparator=uncorrelated_preparator)
        # run_uuid = str(uuid.uuid1()).replace("-", "")
        
        # dirpath = os.path.join(uncorrelated_cfg.root_dir, run_uuid)
        # logger_dir = os.path.join(uncorrelated_cfg.root_dir, run_uuid)
        
        # uncorrelated_trainer, uncorrelated_logger = causal_nf_train.load_trainer(
        #     cfg=uncorrelated_cfg,
        #     dirpath=dirpath,
        #     logger_dir=logger_dir,
        #     include_logger=True,
        #     model_checkpoint=uncorrelated_cfg.train.model_checkpoint,
        #     cfg_early=uncorrelated_cfg.early_stopping,
        #     preparator=uncorrelated_preparator,
        # )
        
        # uncorrelated_trainer.fit(uncorrelated_model, train_dataloaders=uncorrelated_loaders[0], val_dataloaders=uncorrelated_loaders[1])
        # ckpt_name_list = ["last"]
        
        # if uncorrelated_cfg.early_stopping.activate:
        #     ckpt_name_list.append("best")
        
        # for ckpt_name in ckpt_name_list:
        #     for i, loader_i in enumerate(uncorrelated_loaders):
        #         s_name = uncorrelated_preparator.split_names[i]
        #         causal_nf_io.print_info(f"Testing {s_name} split")
        #         uncorrelated_preparator.set_current_split(i)
        #         uncorrelated_model.ckpt_name = ckpt_name
        #         _ = uncorrelated_trainer.test(ckpt_path=ckpt_name, dataloaders=loader_i)
        #         metrics_stats = uncorrelated_model.metrics_stats
        #         metrics_stats["current_epoch"] = uncorrelated_trainer.current_epoch

        # uncorrelated_x_obs = uncorrelated_model.input_scaler.inverse_transform(
        #     uncorrelated_model.model.sample(
        #         (X_uncorrelated.shape[0],)
        #     )['x_obs'], inplace=False
        # )
                
        
        # creating correlated data
        correlated_cfg = copy.deepcopy(cfg)
        correlated_preparator = SCMPreparator.loader(correlated_cfg.dataset)
        pdf_correlated, corr_correlated = correlated_preparator.get_features_all()
        correlated_loaders = correlated_preparator.get_dataloaders(batch_size=correlated_cfg.train.batch_size, num_workers=correlated_cfg.train.num_workers)
        correlated_model = causal_nf_train.load_model(cfg=correlated_cfg, preparator=correlated_preparator)
        run_uuid = str(uuid.uuid1()).replace("-", "")
        dirpath = os.path.join(correlated_cfg.root_dir, run_uuid)
        logger_dir = os.path.join(correlated_cfg.root_dir, run_uuid)
        correlated_trainer, correlated_logger = causal_nf_train.load_trainer(
            cfg=correlated_cfg,
            dirpath=dirpath,
            logger_dir=logger_dir,
            include_logger=True,
            model_checkpoint=correlated_cfg.train.model_checkpoint,
            cfg_early=correlated_cfg.early_stopping,
            preparator=correlated_preparator,
        )
        correlated_trainer.fit(correlated_model, train_dataloaders=correlated_loaders[0], val_dataloaders=correlated_loaders[1])
        ckpt_name_list = ["last"]

        if correlated_cfg.early_stopping.activate:
            ckpt_name_list.append("best")

        for ckpt_name in ckpt_name_list:
            for i, loader_i in enumerate(correlated_loaders):
                s_name = correlated_preparator.split_names[i]
                causal_nf_io.print_info(f"Testing {s_name} split")
                correlated_preparator.set_current_split(i)
                correlated_model.ckpt_name = ckpt_name
                _ = correlated_trainer.test(ckpt_path=ckpt_name, dataloaders=loader_i)
                metrics_stats = correlated_model.metrics_stats
                metrics_stats["current_epoch"] = correlated_trainer.current_epoch

        correlated_x_obs = correlated_model.input_scaler.inverse_transform(
            correlated_model.model.sample(
                (X_correlated.shape[0],)
            )['x_obs'], inplace=False
        )

    


        sign = 'X'

        # save_experiment(uncorrelated_x_obs.detach().numpy(), uncorrelated_cfg.dataset, output_folder)
        # save_experiment(correlated_x_obs.detach().numpy(), correlated_cfg.dataset, output_folder)
        
        plt.rcParams.update(plt.rcParamsDefault)

        corr_correlated = reformat_correlations(cfg.dataset['correlations'])

        # df_uncorrelated = pd.DataFrame(uncorrelated_x_obs.detach().numpy())
        # df_uncorrelated['correlation'] = "uncorrelated"
        # df_uncorrelated['type'] = "observational"

        # df_uncorrelated_base = pd.DataFrame(X_uncorrelated.numpy())
        # df_uncorrelated_base['correlation'] = "originally uncorrelated"
        # df_uncorrelated_base['type'] = "observational"

        df_correlated = pd.DataFrame(correlated_x_obs.detach().numpy())
        # df_correlated['correlation'] =  "correlated: " + corr_correlated[0] # this needs to be changed to work with multiple corr at the same time !!!
        df_correlated['correlation'] = "generated data"
        df_correlated['type'] = "observational"

        df_correlated_base = pd.DataFrame(X_correlated.numpy())
        df_correlated_base['correlation'] = "real data"
        df_correlated_base['type'] = "observational"

        x_int_corr, x_int_corr_true = get_do_samples(correlated_preparator, correlated_model.model, "interventions", correlated_x_obs)
        x_cf_corr, x_cf_corr_true = get_do_samples(correlated_preparator, correlated_model.model, "counterfactual", correlated_x_obs)
        x_ate_corr, x_ate_corr_true = get_do_samples(correlated_preparator, correlated_model.model, "ate", correlated_x_obs)
        

        # x_int_uncorr, x_int_uncorr_true = get_do_samples(uncorrelated_preparator, uncorrelated_model.model, "interventions", uncorrelated_x_obs)
        # x_cf_uncorr, x_cf_uncorr_true = get_do_samples(uncorrelated_preparator, uncorrelated_model.model, "counterfactual", uncorrelated_x_obs)
        # x_ate_uncorr, x_ate_uncorr_true = get_do_samples(uncorrelated_preparator, uncorrelated_model.model, "ate", uncorrelated_x_obs)


        # observed
        df_25p = {}
        for i in range(len(x_int_corr)):
            if x_int_corr[i].name == '25p':
                # int - correlated
                df_25p[f'{sign}{x_int_corr[i].index + 1}'] = pd.DataFrame(
                    x_int_corr[i].values.detach().numpy(), 
                    columns=[
                        f"{sign}{elem + 1}" for elem in range(x_int_corr[i].values.shape[1])
                    ]
                )
                df_25p[f'{sign}{x_int_corr[i].index + 1}']['correlation'] = "generated data"
                df_25p[f'{sign}{x_int_corr[i].index + 1}']['type'] = 'interventional'

                # # int - uncorrelated
                # df_25p[f'{sign}{x_int_uncorr[i].index + 1}'] = pd.DataFrame(
                #     x_int_uncorr[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_int_uncorr[i].values.shape[1])
                #     ]
                # )
                # df_25p[f'{sign}{x_int_uncorr[i].index + 1}']['correlation'] = "uncorrelated"
                # df_25p[f'{sign}{x_int_uncorr[i].index + 1}']['type'] = 'interventional'
                
                # #cf - correlated
                # df_25p[f'{sign}{x_cf_corr[i].index + 1}'] = pd.DataFrame(
                #     x_cf_corr[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_corr[i].values.shape[1])
                #     ]
                # )
                # df_25p[f'{sign}{x_cf_corr[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_25p[f'{sign}{x_cf_corr[i].index + 1}']['type'] = 'counterfactual'

                # #cf - uncorrelated
                # df_25p[f'{sign}{x_cf_uncorr[i].index + 1}'] = pd.DataFrame(
                #     x_cf_uncorr[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_uncorr[i].values.shape[1])
                #     ]
                # )
                # df_25p[f'{sign}{x_cf_uncorr[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_25p[f'{sign}{x_cf_uncorr[i].index + 1}']['type'] = 'counterfactual'
        
        df_50p = {}
        for i in range(len(x_int_corr)):
            if x_int_corr[i].name == '50p':
                # int - correlated
                df_50p[f'{sign}{x_int_corr[i].index + 1}'] = pd.DataFrame(
                    x_int_corr[i].values.detach().numpy(), 
                    columns=[
                        f"{sign}{elem + 1}" for elem in range(x_int_corr[i].values.shape[1])
                    ]
                )
                df_50p[f'{sign}{x_int_corr[i].index + 1}']['correlation'] = "generated data"
                df_50p[f'{sign}{x_int_corr[i].index + 1}']['type'] = 'interventional'

                # # int - uncorrelated
                # df_50p[f'{sign}{x_int_uncorr[i].index + 1}'] = pd.DataFrame(
                #     x_int_uncorr[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_int_uncorr[i].values.shape[1])
                #     ]
                # )
                # df_50p[f'{sign}{x_int_uncorr[i].index + 1}']['correlation'] = "uncorrelated"
                # df_50p[f'{sign}{x_int_uncorr[i].index + 1}']['type'] = 'interventional'
                
                # #cf - correlated
                # df_50p[f'{sign}{x_cf_corr[i].index + 1}'] = pd.DataFrame(
                #     x_cf_corr[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_corr[i].values.shape[1])
                #     ]
                # )
                # df_50p[f'{sign}{x_cf_corr[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_50p[f'{sign}{x_cf_corr[i].index + 1}']['type'] = 'counterfactual'

                # #cf - uncorrelated
                # df_50p[f'{sign}{x_cf_uncorr[i].index + 1}'] = pd.DataFrame(
                #     x_cf_uncorr[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_uncorr[i].values.shape[1])
                #     ]
                # )
                # df_50p[f'{sign}{x_cf_uncorr[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_50p[f'{sign}{x_cf_uncorr[i].index + 1}']['type'] = 'counterfactual'
        
        df_75p = {}
        for i in range(len(x_int_corr)):
            if x_int_corr[i].name == '75p':
                # int - correlated
                df_75p[f'{sign}{x_int_corr[i].index + 1}'] = pd.DataFrame(
                    x_int_corr[i].values.detach().numpy(), 
                    columns=[
                        f"{sign}{elem + 1}" for elem in range(x_int_corr[i].values.shape[1])
                    ]
                )
                df_75p[f'{sign}{x_int_corr[i].index + 1}']['correlation'] = "generated data"
                df_75p[f'{sign}{x_int_corr[i].index + 1}']['type'] = 'interventional'

                # # int - uncorrelated
                # df_75p[f'{sign}{x_int_uncorr[i].index + 1}'] = pd.DataFrame(
                #     x_int_uncorr[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_int_uncorr[i].values.shape[1])
                #     ]
                # )
                # df_75p[f'{sign}{x_int_uncorr[i].index + 1}']['correlation'] = "uncorrelated"
                # df_75p[f'{sign}{x_int_uncorr[i].index + 1}']['type'] = 'interventional'
                
                # #cf - correlated
                # df_75p[f'{sign}{x_cf_corr[i].index + 1}'] = pd.DataFrame(
                #     x_cf_corr[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_corr[i].values.shape[1])
                #     ]
                # )
                # df_75p[f'{sign}{x_cf_corr[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_75p[f'{sign}{x_cf_corr[i].index + 1}']['type'] = 'counterfactual'

                # #cf - uncorrelated
                # df_75p[f'{sign}{x_cf_uncorr[i].index + 1}'] = pd.DataFrame(
                #     x_cf_uncorr[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_uncorr[i].values.shape[1])
                #     ]
                # )
                # df_75p[f'{sign}{x_cf_uncorr[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_75p[f'{sign}{x_cf_uncorr[i].index + 1}']['type'] = 'counterfactual'
        

        # true
        df_25p_true = {}
        for i in range(len(x_int_corr_true)):
            if x_int_corr_true[i].name == '25p':
                # int - correlated
                df_25p_true[f'{sign}{x_int_corr_true[i].index + 1}'] = pd.DataFrame(
                    x_int_corr_true[i].values.detach().numpy(), 
                    columns=[
                        f"{sign}{elem + 1}" for elem in range(x_int_corr_true[i].values.shape[1])
                    ]
                )
                df_25p_true[f'{sign}{x_int_corr_true[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                df_25p_true[f'{sign}{x_int_corr_true[i].index + 1}']['type'] = 'interventional true'

                # # int - uncorrelated
                # df_25p_true[f'{sign}{x_int_uncorr_true[i].index + 1}'] = pd.DataFrame(
                #     x_int_uncorr_true[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_int_uncorr_true[i].values.shape[1])
                #     ]
                # )
                # df_25p_true[f'{sign}{x_int_uncorr_true[i].index + 1}']['correlation'] = "uncorrelated"
                # df_25p_true[f'{sign}{x_int_uncorr_true[i].index + 1}']['type'] = 'interventional true'
                
                # #cf - correlated
                # df_25p_true[f'{sign}{x_cf_corr_true[i].index + 1}'] = pd.DataFrame(
                #     x_cf_corr_true[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_corr_true[i].values.shape[1])
                #     ]
                # )
                # df_25p_true[f'{sign}{x_cf_corr_true[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_25p_true[f'{sign}{x_cf_corr_true[i].index + 1}']['type'] = 'counterfactual'

                # #cf - uncorrelated
                # df_25p_true[f'{sign}{x_cf_uncorr_true[i].index + 1}'] = pd.DataFrame(
                #     x_cf_uncorr_true[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_uncorr_true[i].values.shape[1])
                #     ]
                # )
                # df_25p_true[f'{sign}{x_cf_uncorr_true[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_25p_true[f'{sign}{x_cf_uncorr_true[i].index + 1}']['type'] = 'counterfactual'
        
        df_50p_true = {}
        for i in range(len(x_int_corr_true)):
            if x_int_corr_true[i].name == '50p':
                # int - correlated
                df_50p_true[f'{sign}{x_int_corr_true[i].index + 1}'] = pd.DataFrame(
                    x_int_corr_true[i].values.detach().numpy(), 
                    columns=[
                        f"{sign}{elem + 1}" for elem in range(x_int_corr_true[i].values.shape[1])
                    ]
                )
                df_50p_true[f'{sign}{x_int_corr_true[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                df_50p_true[f'{sign}{x_int_corr_true[i].index + 1}']['type'] = 'interventional true'

                # # int - uncorrelated
                # df_50p_true[f'{sign}{x_int_uncorr_true[i].index + 1}'] = pd.DataFrame(
                #     x_int_uncorr_true[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_int_uncorr_true[i].values.shape[1])
                #     ]
                # )
                # df_50p_true[f'{sign}{x_int_uncorr_true[i].index + 1}']['correlation'] = "uncorrelated"
                # df_50p_true[f'{sign}{x_int_uncorr_true[i].index + 1}']['type'] = 'interventional true'
                
                # #cf - correlated
                # df_50p_true[f'{sign}{x_cf_corr_true[i].index + 1}'] = pd.DataFrame(
                #     x_cf_corr_true[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_corr_true[i].values.shape[1])
                #     ]
                # )
                # df_50p_true[f'{sign}{x_cf_corr_true[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_50p_true[f'{sign}{x_cf_corr_true[i].index + 1}']['type'] = 'counterfactual'

                # #cf - uncorrelated
                # df_50p_true[f'{sign}{x_cf_uncorr_true[i].index + 1}'] = pd.DataFrame(
                #     x_cf_uncorr_true[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_uncorr_true[i].values.shape[1])
                #     ]
                # )
                # df_50p_true[f'{sign}{x_cf_uncorr_true[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_50p_true[f'{sign}{x_cf_uncorr_true[i].index + 1}']['type'] = 'counterfactual'
        
        df_75p_true = {}
        for i in range(len(x_int_corr_true)):
            if x_int_corr_true[i].name == '75p':
                # int - correlated
                df_75p_true[f'{sign}{x_int_corr_true[i].index + 1}'] = pd.DataFrame(
                    x_int_corr_true[i].values.detach().numpy(), 
                    columns=[
                        f"{sign}{elem + 1}" for elem in range(x_int_corr_true[i].values.shape[1])
                    ]
                )
                df_75p_true[f'{sign}{x_int_corr_true[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                df_75p_true[f'{sign}{x_int_corr_true[i].index + 1}']['type'] = 'interventional true'

                # # int - uncorrelated
                # df_75p_true[f'{sign}{x_int_uncorr_true[i].index + 1}'] = pd.DataFrame(
                #     x_int_uncorr_true[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_int_uncorr_true[i].values.shape[1])
                #     ]
                # )
                # df_75p_true[f'{sign}{x_int_uncorr_true[i].index + 1}']['correlation'] = "uncorrelated"
                # df_75p_true[f'{sign}{x_int_uncorr_true[i].index + 1}']['type'] = 'interventional true'
                
                # #cf - correlated
                # df_75p_true[f'{sign}{x_cf_corr_true[i].index + 1}'] = pd.DataFrame(
                #     x_cf_corr_true[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_corr_true[i].values.shape[1])
                #     ]
                # )
                # df_75p_true[f'{sign}{x_cf_corr_true[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_75p_true[f'{sign}{x_cf_corr_true[i].index + 1}']['type'] = 'counterfactual'

                # #cf - uncorrelated
                # df_75p_true[f'{sign}{x_cf_uncorr_true[i].index + 1}'] = pd.DataFrame(
                #     x_cf_uncorr_true[i].values.detach().numpy(), 
                #     columns=[
                #         f"{sign}{elem + 1}" for elem in range(x_cf_uncorr_true[i].values.shape[1])
                #     ]
                # )
                # df_75p_true[f'{sign}{x_cf_uncorr_true[i].index + 1}']['correlation'] = "correlated: " + corr_correlated[0]
                # df_75p_true[f'{sign}{x_cf_uncorr_true[i].index + 1}']['type'] = 'counterfactual'
        
        intervened_var = 'X4'

        df_combined = pd.concat([
            # df_uncorrelated, 
            df_correlated, 
            # df_uncorrelated_base, 
            df_correlated_base
            # df_75p[intervened_var], 
            # df_75p_true[intervened_var] 
        ], axis=0)
        df_combined.columns = [f"{sign}{i}" for i in range(1, correlated_x_obs.shape[1] + 1)] + ['correlation'] + ['type']

        sign = 'U'

        df_X_obs = pd.DataFrame(
            np.load('/Users/georgivitanov/Desktop/Thesis/causal-flows-assumptions/chain5_u_0_2_0_samples.npy'), 
            columns=[f"{sign}{i}" for i in range(1, correlated_x_obs.shape[1] + 1)]
        )
        df_X_obs['type'] = 'real data - u with 0 correlation'

        df_X = pd.DataFrame(
            np.load('/Users/georgivitanov/Desktop/Thesis/causal-flows-assumptions/chain5_0_2_075_u_samples.npy'), 
            columns=[f"{sign}{i}" for i in range(1, correlated_x_obs.shape[1] + 1)]
        )
        df_X['type'] = 'real data - u with 0.75 correlation'

        # sns_plot = sns.pairplot(pd.concat([df_X, df_X_obs], axis=0), hue='type')
        sns_plot = sns.pairplot(df_combined, hue='correlation')
        sns_plot.map_offdiag(sns.kdeplot, levels=4, color=".1")

        sns_plot.figure.subplots_adjust(top=0.95)
        # sns_plot.figure.suptitle(
        #     f"{cfg.dataset['name']}, {corr_correlated[0]}, \
        #           type: {df_combined['type'].unique()[0]}, \
        #             correlation: {df_combined['correlation'][0].unique()[0]}"
        # ) 
        # sns_plot.figure.suptitle(f"{cfg.dataset['name']}, {corr_correlated[0]}, type: {df_combined['type'].unique()[0]}, correlation: {df_combined['correlation'][0].unique()[0]}, do({intervened_var}=75p)") 
        # pdb.set_trace()
        # fig_name = f"{cfg.dataset['name']}_{corr_correlated[0]}_int_{intervened_var}.png"
        sns_plot.figure.suptitle(f"corr(U1, U3) = 0.0, inner layer [ 512 ] x 1, 5000 samples")
        fig_name =f"Original code with corr(U1, U3) = 0"
        sns_plot.figure.savefig(
            os.path.join(output_folder, fig_name)
        )

      





    output_folder = create_output_folder()

    run_experiments(cfg, output_folder)

                
    




print("Hello World")