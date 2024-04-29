#!/bin/bash

declare -a arr=('kl_distance' 'log_prob' 
    'log_prob_p' 'log_prob_true' 'loss' 'mmd_int_x1=25p' 
    'mmd_int_x1=50p' 'mmd_int_x1=75p' 'mmd_int_x2=25p' 
    'mmd_int_x2=50p' 'mmd_int_x2=75p' 'mmd_int_x3=25p' 
    'mmd_int_x3=50p' 'mmd_int_x3=75p' 'mmd_obs' 'mse_cf_x1=25p' 
    'mse_cf_x1=50p' 'mse_cf_x1=75p' 'mse_cf_x2=25p' 'mse_cf_x2=50p' 
    'mse_cf_x2=75p' 'mse_cf_x3=25p' 'mse_cf_x3=50p' 'mse_cf_x3=75p' 
    'rmse_ate_x1=25_50' 'rmse_ate_x1=25_75' 'rmse_ate_x1=50_75' 
    'rmse_ate_x2=25_50' 'rmse_ate_x2=25_75' 'rmse_ate_x2=50_75' 
    'rmse_ate_x3=25_50' 'rmse_ate_x3=25_75' 'rmse_ate_x3=50_75' 
    'rmse_cf_x1=25p' 'rmse_cf_x1=50p' 'rmse_cf_x1=75p' 
    'rmse_cf_x2=25p' 'rmse_cf_x2=50p' 'rmse_cf_x2=75p' 
    'rmse_cf_x3=25p' 'rmse_cf_x3=50p' 'rmse_cf_x3=75p')


for metric in "${arr[@]}"
 do

    python aggregate_results_temp.py \
     --metric $metric \
      --affected_var 1 2 

done


