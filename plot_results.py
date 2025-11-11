
import utilities.plots_utils as plu

def main():

    # Prepare stats
    # files_name_to_process = ['MB_rn10_rn18_GCM',
    #                         'MB_rn10_rn18_LCM',
    #                         'NL_all_arch_GCM',
    #                         'NL_all_arch_LCM',
    #                         'NO_all_arch_GCM', 
    #                         'NO_all_arch_LCM']

    
    seeds = [42,5,997]
    files_name_to_process = ['model_results']
    for f in files_name_to_process:
        print(f'Processing {f}...')
        plu.prepare_csv_stats(csv_path = f'.\Results_FINAL\Results\{f}.csv', # f'.\Results\!!_FINAL_results\{f}.csv'
                              columns_to_keep = ['model_name','base_model', 'mode'],
                              columns_to_avg = ['MSE', 'MAE', 'R2'],
                              group_by = ['model_name','mode'],
                              seeds = seeds,
                              output_path = f'.\Results_FINAL\Results\!_tables\{f}_stats.csv',
        )
        print(f'{f}_stats saved in CSV format. Finished processing {f}.\n\n')

    
    # Plot average loss curves
    print('Plotting average loss curves...')
    print('Noiseless systems:')
    base_system = '2-0'
    resnet_versions = ['Rn10', 'Rn12','Rn14','Rn16','Rn18']
    # main_folder_paths = ['.\Results\Results-GCM-NL', '.\Results\Results-LCM-NL']
    main_folder_paths = ['.\Results_FINAL\Results']

    for rv in resnet_versions:
        systems = [f'{rv}-{base_system}']
        rn_v = rv if len(rv) == 1 else f'{rv[0]}-{rv[-1]}'
        plu.plot_average_loss_curves(main_folder_paths = main_folder_paths,
                                    systems = systems,
                                    seeds = seeds,
                                    output_path = f'.\Results_FINAL\Results\!_Figs\{base_system}-{rv}-loss_curves.png')
    print('Done. \n\n')
    print('Noiseless systems:')
    base_system = '2-1'
    resnet_versions = ['Rn10', 'Rn12','Rn14','Rn16','Rn18']
    # main_folder_paths = ['.\Results\Results-GCM-NL', '.\Results\Results-LCM-NL']
    main_folder_paths = ['.\Results_FINAL\Results']

    for rv in resnet_versions:
        systems = [f'{rv}-{base_system}']
        rn_v = rv if len(rv) == 1 else f'{rv[0]}-{rv[-1]}'
        plu.plot_average_loss_curves(main_folder_paths = main_folder_paths,
                                    systems = systems,
                                    seeds = seeds,
                                    output_path = f'.\Results_FINAL\Results\!_Figs\{base_system}-{rv}-loss_curves.png')
    print('Done. \n\n')


    print('Noisy systems with n8e5 noise')
    base_system = '2-1-n8e5'
    # main_folder_paths = ['.\Results\Results-GCM-NO', '.\Results\Results-LCM-NO']
    for rv in resnet_versions:
        systems = [f'{rv}-{base_system}']
        rn_v = rv if len(rv) == 1 else f'{rv[0]}-{rv[-1]}'
        plu.plot_average_loss_curves(main_folder_paths = main_folder_paths,
                                    systems = systems,
                                    seeds = seeds,
                                    output_path = f'.\Results_FINAL\Results\!_Figs\{base_system}-{rn_v}-loss_curves.png')
    print('Done. \n\n')

    print('Noisy systems with n2e5 noise')
    base_system = '2-1-n2e5'
    # main_folder_paths = ['.\Results\Results-GCM-NO', '.\Results\Results-LCM-NO']
    for rv in resnet_versions:
        systems = [f'{rv}-{base_system}']
        rn_v = rv if len(rv) == 1 else f'{rv[0]}-{rv[-1]}'
        plu.plot_average_loss_curves(main_folder_paths = main_folder_paths,
                                    systems = systems,
                                    seeds = seeds,
                                    output_path = f'.\Results_FINAL\Results\!_Figs\{base_system}-{rn_v}-loss_curves.png')
    print('Done. \n\n')

    print('Big systems with n5e5 noise')
    base_system = '3-2'
    resnet_versions = ['Rn10','Rn18']
    # main_folder_paths = ['.\Results\Results-GCM-NO-5-3-2-Rn10-Rn18', '.\Results\Results-LCM-NO-5-3-2-Rn10-Rn18']
    for rv in resnet_versions:
        systems = [f'{rv}-{base_system}']
        rn_v = rv if len(rv) == 1 else f'{rv[0]}-{rv[-1]}'
        plu.plot_average_loss_curves(main_folder_paths = main_folder_paths,
                                    systems = systems,
                                    seeds = seeds,
                                    output_path = f'.\Results_FINAL\Results\!_Figs\{base_system}-{rn_v}-loss_curves.png')
    print('Done. \n\n')
    
    #Plot results for noiseless case
    print('Plotting MSE vs MAE + R2 for all systems')
    scale = 750
    # csv_file = './Results/!!_FINAL_results/stats/NL_all_arch_all_stats.csv'
    csv_file = './Results_FINAL/Results/!_tables/model_results_stats.csv'
    plu.exp1_plot_mse_mae_r2(csv_path = csv_file, 
                            system_name='-2-0',
                            output_path='.\Results_FINAL\Results\!_Figs',
                            r2_scale=scale)

    plu.exp1_plot_mse_mae_r2(csv_path = csv_file, 
                            system_name='-2-1',
                            output_path='.\Results_FINAL\Results\!_Figs',
                            r2_scale=scale)
    
    # csv_file = './Results/!!_FINAL_results/stats/NO_all_arch_all_stats.csv'
    plu.exp1_plot_mse_mae_r2(csv_path = csv_file, 
                            system_name='-2-1-n8e5',
                            output_path='.\Results_FINAL\Results\!_Figs',
                            r2_scale=scale)

    plu.exp1_plot_mse_mae_r2(csv_path = csv_file, 
                            system_name='-2-1-n2e5',
                            output_path='.\Results_FINAL\Results\!_Figs',
                            r2_scale=scale)

    # csv_file = './Results/!!_FINAL_results/stats/MB_all_stats.csv'
    plu.exp1_plot_mse_mae_r2(csv_path = csv_file, 
                            system_name='-3-2',
                            output_path='.\Results_FINAL\Results\!_Figs',
                            r2_scale=scale)

    print('Done. \n\n')

    # # MSE chosen element of the predicted vector for all systems
    # main_folder_paths = ['.\Results\Results-GCM-NL', 
    #                      '.\Results\Results-LCM-NL',
    #                      '.\Results\Results-GCM-NO', 
    #                      '.\Results\Results-LCM-NO',
    #                      '.\Results\Results-GCM-NO-5-3-2-Rn10-Rn18', 
    #                      '.\Results\Results-LCM-NO-5-3-2-Rn10-Rn18']
    
    devices = ['2-0', '2-1', '2-1-n8e5', '2-1-n2e5']
    resnet_versions = ['Rn10', 'Rn12', 'Rn14', 'Rn16', 'Rn18']
    systems = [f'{rv}-{device}' for rv in resnet_versions for device in devices]
    systems.append('Rn10-3-2')
    systems.append('Rn18-3-2')

    main_folder_paths = ['.\Results_FINAL\Results']
    for element_index in range(2):
        plu.plot_mse_chosen_element(main_folder_paths = main_folder_paths,
                               systems = systems,
                               seeds = seeds,
                               element_index = element_index,
                               interaction_type = 'dd',
                               output_path = f'.\Results_FINAL\Results\!_Figs\mse_chosen_element_dd_{element_index}.png')



    
    devices = ['2-1', '2-1-n8e5', '2-1-n2e5']
    resnet_versions = ['Rn10', 'Rn12', 'Rn14', 'Rn16', 'Rn18']
    systems = [f'{rv}-{device}' for rv in resnet_versions for device in devices]
    # systems.append('Rn10-3-2')
    # systems.append('Rn18-3-2')

    main_folder_paths = ['.\Results_FINAL\Results']

    for element_index in [0,1,3,5]:
        plu.plot_residuals_chosen_element(main_folder_paths = main_folder_paths,
                                     systems = systems,
                                     seeds = seeds,
                                     element_index = element_index,
                                     interaction_type = 'dd',   
                                     output_path = f'.\Results_FINAL\Results\!_Figs/residuals_chosen_element_dd_{element_index}.png')


def plot_capacitance_matrix_elements():
    seeds = [42,5,997]
    main_folder_paths = ['.\Results_FINAL\Results']
    devices = ['2-1', '2-1-n8e5', '2-1-n2e5']
    resnet_versions = ['Rn10', 'Rn12', 'Rn14', 'Rn16', 'Rn18']
    systems = [f'{rv}-{device}' for rv in resnet_versions for device in devices]
    plu.plot_capacitance_matrix_elements(main_folder_paths = main_folder_paths,
                                        systems = systems,
                                        seeds = seeds,
                                        output_path = f'.\Results_FINAL\Results\!_Figs/capacitance_matrix_elements_2-1.png')
    

    devices = ['3-2']
    resnet_versions = ['Rn10', 'Rn18']
    systems = [f'{rv}-{device}' for rv in resnet_versions for device in devices]
    plu.plot_capacitance_matrix_elements(main_folder_paths = main_folder_paths,
                                         systems = systems,
                                        seeds = seeds,
                                        output_path = f'.\Results_FINAL\Results\!_Figs/capacitance_matrix_elements_3-2.png')
if __name__ == '__main__':
    main()
    # plot_capacitance_matrix_elements()