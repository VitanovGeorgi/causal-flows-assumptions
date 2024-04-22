#!/bin/bash
PATH="/netscratch/gvitanov/envs/cnf"

$PATH/bin/python main.py --config_file grids/causal_nf/ablation_u_x/base/configs/1/config_5.yaml --wandb_mode offline --wandb_group ablation_u_x --project Test; 

$PATH/bin/python main.py --config_file grids/causal_nf/ablation_u_x/base/configs/1/config_6.yaml --wandb_mode offline --wandb_group ablation_u_x --project Test; 

$PATH/bin/python main.py --config_file grids/causal_nf/ablation_u_x/base/configs/1/config_7.yaml --wandb_mode offline --wandb_group ablation_u_x --project Test; 

$PATH/bin/python main.py --config_file grids/causal_nf/ablation_u_x/base/configs/1/config_8.yaml --wandb_mode offline --wandb_group ablation_u_x --project Test; 

