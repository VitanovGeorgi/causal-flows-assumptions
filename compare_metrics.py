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

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import seaborn


import numpy as np
import pandas as pd


os.environ["WANDB_NOTEBOOK_NAME"] = "name_of_the_notebook"

args_list, args = causal_nf_config.parse_args()

load_model = isinstance(args.load_model, str)
if load_model:
    causal_nf_io.print_info(f"Loading model: {args.load_model}")

config = causal_nf_config.build_config(
    config_file=args.config_file,
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

    def save_experiment(changing_data, changing_cfg, output_folder: str = None):
        if not output_folder is None and args.save_metrics_mode == "enabled":
            plt.rcParams.update(plt.rcParamsDefault)
            sns_plot = seaborn.pairplot(
                    pd.DataFrame(changing_data.numpy())
                )
            sns_plot.map_offdiag(seaborn.kdeplot, levels=4, color=".1")
            
            sns_plot.add_legend(
                title=f"{changing_cfg['name']} Us\n {changing_cfg['correlations']}"
            )
            # pdb.set_trace()
            fig_name = f"{changing_cfg['name']}_{changing_cfg['correlations']}.png"
            sns_plot.figure.savefig(
                os.path.join(output_folder, fig_name)
            )



    def run_experiments(cfg, output_folder: str = None):
        cov_epsilon = torch.rand(size=(5, 5)) / 10
        uncorrelated_cfg = copy.deepcopy(cfg.dataset)

        uncorrelated_cfg['correlations'] = None
        preparator = SCMPreparator.loader(uncorrelated_cfg)
        X_full, U_full = preparator.prepare_data(True)
        pdf_1, _ = preparator.get_features_all()
        # full_data = preparator.get_features_train()

        steps = cfg.dataset['steps_correlations']

        changing_cfg = copy.deepcopy(cfg.dataset)
        mmd_historic = list()

        for i in range(steps + 1):
            # pdb.set_trace()
            if not cfg.dataset['correlations'] is None:
                for elem1, elem2 in zip(changing_cfg['correlations'], cfg.dataset['correlations']):
                    elem1[-1] = round(elem2[-1] * i / steps, 4)

            try:
                changing_preparator = SCMPreparator.loader(changing_cfg)
                X_changing, U_changing = changing_preparator.prepare_data(True)
                pdf_2, corr_2 = changing_preparator.get_features_all()
            except:
                mmd_historic.append(None)
                print(f"Correlation of {corr_2} makes the matrix not positive definite, step {i}")
                continue

            mmd_value = maximum_mean_discrepancy(U_changing, U_full)
            # mmd_value = jensen_shannon_divergence(pdf_1, pdf_2)

            if output_folder is not None:
                save_experiment(U_changing, changing_cfg, output_folder)
                        
            mmd_historic.append(mmd_value.item())
            print(mmd_value.item())
            
        print(f"Int {sum(filter(None, mmd_historic)) / steps}")
        plt.clf() # clean the plt from the previous plots, o.w. they're being saved
        plt.rcParams.update(plt.rcParamsDefault)
        plt.plot(np.linspace(0, 1, len(mmd_historic)), mmd_historic)
        plt.xlabel("Correlation")
        plt.ylabel("MMD")
        try:
            plt.title(
                f"SEM {cfg.dataset['name']}, means {pdf_1.loc.tolist()}, variances {torch.diagonal(pdf_1.covariance_matrix).tolist()}, correlations {corr_2}", 
                fontsize=8
            )
        except:
            pass
        if output_folder is not None:
            plt.savefig(
                os.path.join(output_folder, f"metrics.png")
            )
        else:
            plt.show()






    output_folder = create_output_folder()

    run_experiments(cfg, output_folder)

                
    




print("Hello World")