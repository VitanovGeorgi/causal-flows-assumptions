echo $PATH

PATH="/netscratch/gvitanov/envs/cnf"


$PATH/bin/python main.py --config_file grids/causal_nf/ablation_u_x/base/configs/1/config_1.yaml --wandb_mode offline --wandb_group ablation_u_x --project Test; 

$PATH/bin/python main.py --config_file grids/causal_nf/ablation_u_x/base/configs/1/config_2.yaml --wandb_mode offline --wandb_group ablation_u_x --project Test; 

$PATH/bin/python main.py --config_file grids/causal_nf/ablation_u_x/base/configs/1/config_3.yaml --wandb_mode offline --wandb_group ablation_u_x --project Test; 

$PATH/bin/python main.py --config_file grids/causal_nf/ablation_u_x/base/configs/1/config_4.yaml --wandb_mode offline --wandb_group ablation_u_x --project Test; 

