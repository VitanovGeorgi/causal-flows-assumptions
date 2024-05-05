#!/bin/bash

declare -a arr=('kl_distance' 'log_prob' 
    'log_prob_p' 'log_prob_true' 'loss' 'mmd_int_x1' 
    'mmd_int_x2' 'mmd_int_x3' 'mmd_obs' 
    'mse_cf_x1' 'mse_cf_x2' 'mse_cf_x3'
    'rmse_ate_x1' 'rmse_ate_x2' 'rmse_ate_x3' 
    'rmse_cf_x1' 'rmse_cf_x2' 'rmse_cf_x3')


for metric in "${arr[@]}"
 do

    python plot_aggregated_results.py \
    --source=output_causal_nf/aggregated_results/results_chain \
     --metric $metric \
    #   --affected_var 2,3 


done


