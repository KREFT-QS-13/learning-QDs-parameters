import argparse
import sys, time
import numpy as np
import multiprocessing as mp

from utilities.config import K as config_K, set_global_K
import utilities.utils as u

def main():
    x_vol = np.linspace(0, 0.05, 500)
    y_vol = np.linspace(0, 0.05, 500)

    x_vol_range = (x_vol[-1], len(x_vol))
    y_vol_range = (y_vol[-1], len(y_vol))

    ks = None

    parser = argparse.ArgumentParser(description="Generating a dataset.")

    parser.add_argument('--N', type=np.int32, default=250, 
                        help='The size of a single batch of data. Default vaule 250.')
    
    parser.add_argument('--R', type=np.int32, default=1, 
                        help='The number of batches to to generate. Default vaule 1.')
    
    parser.add_argument('--K', type=np.int32, default=config_K, 
                        help='The number of quantum dots in the system. Default vaule 2.')
    #TODO: Later add noise parameter
    # parser.add_argument('--Noise', type=, help='')

    args = parser.parse_args()
    N = args.N
    R = args.R
    K = args.K

    set_global_K(K)

    for r in range(R):
        print(f"Batch number: {r+1}/{R}.")
        main_start = time.time()
        u.create_paths(K)
        # Prepare arguments for multiprocessing
        pool_args = [(K, x_vol, y_vol, ks, i, N) for i in range(N)]
        
        # Create a multiprocessing pool
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(u.generate_datapoint, pool_args)

        suc = 0
        for i, result in enumerate(results):
            if result is not None:
                suc += 1
                C_DD, C_DG, ks, cuts, x_vol, y_vol, fig = result
                u.save_datapoints(K, C_DD, C_DG, ks, x_vol_range, y_vol_range, cuts, fig)
                print(f"Successfully generated datapoints: {suc}/{N} ({i+1}/{N}).\n\n")

        final_time = round(np.abs(main_start-time.time()),3)
        print(f"\nTotal time: {final_time}[s] -> {(final_time/N):.3f}[s] per datapoint.")    
        print(f"Successfully generated datapoints: {suc}/{N}.\n\n")
        
        if R>1:
            print("Rest for 2 mintues, to decreser the tmep. of the CPU.")
            time.sleep(120) # two mintes break
if __name__ == "__main__":
    main()
