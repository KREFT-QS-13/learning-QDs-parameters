import argparse
import sys, time
import numpy as np
import multiprocessing as mp
import importlib

import utilities.config as c
import utilities.utils as u

def main():
    x_vol = np.linspace(0, 0.05, c.RESOLUTION)
    y_vol = np.linspace(0, 0.05, c.RESOLUTION)

    x_vol_range = (x_vol[-1], len(x_vol))
    y_vol_range = (y_vol[-1], len(y_vol))

    ks = None

    parser = argparse.ArgumentParser(description="Generating a dataset.")

    parser.add_argument('-N', type=np.int32, default=250, 
                        help='The size of a single batch of data. Default vaule 250.')
    
    parser.add_argument('-R', type=np.int32, default=1, 
                        help='The number of batches to to generate. Default vaule 1.')
    
    parser.add_argument('-K', type=np.int32, default=2, 
                        help='The number of quantum dots in the system. Default vaule 2.')
   
    parser.add_argument('-S', type=int, default=0, 
                        help='The number of sensors in the system.')

    parser.add_argument('--device', type=list, default=np.ones((1,2)), 
                        help='The device to generate the dataset for.')


    args = parser.parse_args()
    N_batch = args.N
    R = args.R
    K = args.K

    if args.S>1 and args.device is None:
        raise ValueError("The device and the number of sensors must be provided when noise is used.")
    else:
        device = np.array(args.device)
        S = args.S
        N_dots = len(u.get_dots_indices(device))
        K = N_dots + S
        
    config_tuple = (K, N_dots, S)
    c.validate_state(*config_tuple)
    
    for r in range(R):
        print(f"Batch number: {r+1}/{R}.")
        main_start = time.time()
        u.create_paths(config_tuple)
        
        # Prepare arguments for multiprocessing
        pool_args = [(x_vol, y_vol, ks, device, i, N_batch, config_tuple) for i in range(N_batch)]
        
        # Create a multiprocessing pool
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(u.generate_datapoint, pool_args)
        
        suc = 0
        for i, result in enumerate(results):
            if result is not None:
                suc += 1
            
                C_DD, C_DG, ks, cut, x_vol, y_vol, fig, fig_gradient, device = result
                u.save_datapoints(config_tuple, C_DD, C_DG, ks, x_vol_range, 
                                  y_vol_range, cut, fig, fig_gradient, device)
                print(f"Successfully generated datapoints: {suc}/{N_batch} ({i+1}/{N_batch}).\n\n")

        final_time = round(np.abs(main_start-time.time()),3)
        print(f"\nTotal time: {final_time}[s] -> {(final_time/N_batch):.3f}[s] per datapoint.")    
        print(f"Successfully generated datapoints: {suc}/{N_batch}.\n\n")
        
        if R>1:
            print("Rest for 1 mintues, to decreser the tmep. of the CPU.")
            time.sleep(60) # two mintes break
        
        print(f"Batch number finished: {r+1}/{R}.")
if __name__ == "__main__":
    main()
