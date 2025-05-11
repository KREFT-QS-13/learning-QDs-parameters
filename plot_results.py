
import utilities.plots_utils as plu

def main():

    # Prepare stats
    files_name_to_process = ['MB_rn10_rn18_GCM', 
                            'NL_all_arch_GCM', 
                            'NO_all_arch_GCM', 
                            'NO_all_arch_LCM']
    seeds = [42,5,997, 89]

    for f in files_name_to_process:
        print(f'Processing {f}...')
        plu.prepare_csv_stats(csv_path = f'.\Results\!!_FINAL_results\{f}.csv',
                              columns_to_keep = ['model_name','base_model', 'mode'],
                              columns_to_avg = ['MSE', 'MAE', 'R2'],
                              group_by = 'model_name',
                              seed_column = 'seed',
                              seeds = seeds,
                              output_path = f'.\Results\!!_FINAL_results\stats\{f}_stats.csv',
        )
        print(f'{f}_stats saved in CSV format. Finished processing {f}.\n\n')

    
    # Plot results for noiseless case
    # csv_file = '.\Results\!_model_results-noiseless.csv'
    # plu.exp1_plot_mse_mae_r2(csv_path = csv_file, 
    #                         system_name='-2-0',
    #                         output_path='.\Results\Figs',
    #                         r2_scale=10)

    # plu.exp1_plot_mse_mae_r2(csv_path = csv_file, 
    #                         system_name='-2-1',
    #                         output_path='.\Results\Figs',
    #                         r2_scale=10)
if __name__ == '__main__':
    main()