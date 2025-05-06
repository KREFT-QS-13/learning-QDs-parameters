
import utilities.plots_utils as plu

def main():
    csv_file = '.\Results\!_model_results-noiseless.csv'
    plu.exp1_plot_mse_mae_r2(csv_path = csv_file, 
                            system_name='-2-0',
                            output_path='.\Results\Figs',
                            r2_scale=10)

    plu.exp1_plot_mse_mae_r2(csv_path = csv_file, 
                            system_name='-2-1',
                            output_path='.\Results\Figs',
                            r2_scale=10)
if __name__ == '__main__':
    main()