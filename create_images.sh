#!/bin/bash

declare -a arr=('kl_distance' 'log_prob' 
    'log_prob_p' 'log_prob_true' 'log_prob_diff' 
    'loss' 'mmd_int_x1' 'mmd_int_x2' 
    'mmd_int_x3' 'mmd_obs' 'mse_cf_x1' 
    'mse_cf_x2' 'mse_cf_x3' 'rmse_ate_x1' 
    'rmse_ate_x2' 'rmse_ate_x3' 'rmse_cf_x1' 
    'rmse_cf_x2' 'rmse_cf_x3')


for metric in "${arr[@]}"
 do

    python plot_aggregated_results_v2.py \
    --source=output_causal_nf/aggregated_results/results_num_samples_chain_3_2024-07-12 \
    --target=output_aggregated_results/chain_3_num_samples \
    --metric $metric \
    --split test 
    #   --affected_var 1,2 1,3 1,4 1,2,1,3 1,2,1,4 1,3,1,4 1,2,1,3,1,4

done


