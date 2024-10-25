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

args.save_metrics_mode = "enabled" # "enabled" or "disabled" comment out

load_model = isinstance(args.load_model, str)
if load_model:
    causal_nf_io.print_info(f"Loading model: {args.load_model}")

config = causal_nf_config.build_config(
    config_file= "grids/causal_nf/comparison_distributions/model_1.yaml", # args.config_file,
    args_list=args_list,
    config_default_file=args.config_default_file,
)

cfg_1 = copy.deepcopy(cfg)

config_2 = causal_nf_config.build_config(
    config_file= "grids/causal_nf/comparison_distributions/model_2.yaml", # args.config_file,
    args_list=args_list,
    config_default_file=args.config_default_file,
)

cfg_2 = copy.deepcopy(cfg)

cfg = cfg_1

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


    def run_experiments(cfg_1, cfg_2, hidden_confounder_idx: int, output_folder: str = None):
        run = wandb.init(
            mode=args.wandb_mode,
            group=args.wandb_group,
            project=args.project,
            config=config,
        )

        # creating model_1 data (usually the one with hidden confounder in it!)
        model_1_preparator = SCMPreparator.loader(cfg_1.dataset)
        X_1, U_1 = model_1_preparator.prepare_data(True)
        pdf_1, corr_1 = model_1_preparator.get_features_all()
        # the loaders still contain dataset.X/U_full
        model_1_loaders = model_1_preparator.get_dataloaders(
            batch_size=cfg_1.train.batch_size, num_workers=cfg_1.train.num_workers
        )

        model_1 = causal_nf_train.load_model(cfg=cfg_1, preparator=model_1_preparator)
        run_uuid = str(uuid.uuid1()).replace("-", "")
        
        dirpath = os.path.join(cfg_1.root_dir, run_uuid)
        logger_dir = os.path.join(cfg_1.root_dir, run_uuid)
        
        model_1_trainer, model_1__logger = causal_nf_train.load_trainer(
            cfg=cfg_1,
            dirpath=dirpath,
            logger_dir=logger_dir,
            include_logger=True,
            model_checkpoint=cfg_1.train.model_checkpoint,
            cfg_early=cfg_1.early_stopping,
            preparator=model_1_preparator,
        )
        
        model_1_trainer.fit(model_1, train_dataloaders=model_1_loaders[0], val_dataloaders=model_1_loaders[1])
        ckpt_name_list = ["last"]
        
        if cfg_1.early_stopping.activate:
            ckpt_name_list.append("best")
        
        for ckpt_name in ckpt_name_list:
            for i, loader_i in enumerate(model_1_loaders):
                s_name = model_1_preparator.split_names[i]
                causal_nf_io.print_info(f"Testing {s_name} split")
                model_1_preparator.set_current_split(i)
                model_1.ckpt_name = ckpt_name
                _ = model_1_trainer.test(ckpt_path=ckpt_name, dataloaders=loader_i)
                metrics_stats = model_1.metrics_stats
                metrics_stats["current_epoch"] = model_1_trainer.current_epoch

        X_1_obs = model_1.input_scaler.inverse_transform(
            model_1.model.sample((X_1.shape[0],))['x_obs'], inplace=False
        )

        U_1_obs = model_1.input_scaler.inverse_transform(
            model_1.model.sample((U_1.shape[0],))['u_obs'], inplace=False
        )
        

        # save_experiment(uncorrelated_x_obs.detach().numpy(), uncorrelated_cfg.dataset, output_folder)
        # save_experiment(correlated_x_obs.detach().numpy(), correlated_cfg.dataset, output_folder)
        
        plt.rcParams.update(plt.rcParamsDefault)

        
        df_x1_obs = pd.DataFrame.from_dict({
            'x1': X_1_obs[:, 0].detach().numpy(),
            'x2': X_1_obs[:, 1].detach().numpy(),
            'x3': X_1_obs[:, 2].detach().numpy(),
            # 'x4': X_1_obs[:, 3].detach().numpy(),
            'type': f'x_obs[1:4] ~ p_{cfg_1.dataset["name"]}'
        })
        
        df_x1_true = pd.DataFrame.from_dict({
            'x1': X_1[:, 0].detach().numpy(),
            'x2': X_1[:, 1].detach().numpy(),
            'x3': X_1[:, 2].detach().numpy(),
            # 'x4': X_1[:, 3].detach().numpy(),
            'type': f'x_true ~ p_{cfg_1.dataset["name"]}'
        })

        df_u1_obs = pd.DataFrame.from_dict({
            'u1': U_1_obs[:, 0].detach().numpy(),
            'u2': U_1_obs[:, 1].detach().numpy(),
            'u3': U_1_obs[:, 2].detach().numpy(),
            'type': 'u_obs'
        })

        df_u1_true = pd.DataFrame.from_dict({
            'u1': U_1[:, 0].detach().numpy(),
            'u2': U_1[:, 1].detach().numpy(),
            'u3': U_1[:, 2].detach().numpy(),
            'type': 'u_true'
        })

        sns_plot_1 = sns.pairplot(pd.concat([df_x1_obs, df_x1_true], axis=0), hue='type')
        sns_plot_1.map_offdiag(sns.kdeplot, levels=4, color=".1")

        sns_plot_1.figure.subplots_adjust(top=0.95) 
        sns_plot_1.figure.suptitle(f"{cfg_1.dataset['name']} v. {cfg_2.dataset['name']}, {cfg.dataset['num_samples']} samples")
        fig_name =f"{cfg_1.dataset['name']}_{cfg_2.dataset['name']}_{cfg.dataset['num_samples']}_1.png"
        sns_plot_1.figure.savefig(
            os.path.join(output_folder, fig_name)
        )

        sns_plot_2 = sns.pairplot(pd.concat([df_u1_obs, df_u1_true], axis=0), hue='type')
        sns_plot_2.map_offdiag(sns.kdeplot, levels=4, color=".1")

        sns_plot_2.figure.subplots_adjust(top=0.95) 
        sns_plot_2.figure.suptitle(f"{cfg_1.dataset['name']} v. {cfg_2.dataset['name']}, {cfg.dataset['num_samples']} samples")
        fig_name =f"{cfg_1.dataset['name']}_{cfg_2.dataset['name']}_{cfg.dataset['num_samples']}_2.png"
        sns_plot_2.figure.savefig(
            os.path.join(output_folder, fig_name)
        )


      # kl (32.15525817871094, 909.1610717773438)
      # log (-9.346009254455566, -8.07353687286377)





    output_folder = create_output_folder()

    run_experiments(cfg_1, cfg_2, 3, output_folder)

                
    




print("Hello World")