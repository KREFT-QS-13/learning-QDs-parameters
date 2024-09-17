import argparse
import numpy as np
import sys, time

import utilities as u

def main():
    x_vol = np.linspace(0, 0.1, 500)
    y_vol = np.linspace(0, 0.1, 500)

    x_vol_range = (x_vol[-1], len(x_vol))
    y_vol_range = (y_vol[-1], len(y_vol))

    ks = None

    parser = argparse.ArgumentParser(description="Generating a dataset.")

    parser.add_argument('--N', type=np.int32, default=250, 
                        help='The number of datapoints to generate. Default vaule 250.')
    
    parser.add_argument('--K', type=np.int32, default=2, 
                        help='The number of quantum dots in the system. Default vaule 2.')
    #TODO: Later add noise parameter
    # parser.add_argument('--Noise', type=, help='')

    args = parser.parse_args()
    suc = 0
    N = args.N
    K = args.K

    main_start = time.time()
    u.create_paths(K)
    for i in range(N):
        print(f"Generating datapoint {i+1}/{N}:")
        # C_DD, C_DG, ks, cuts, x, y, csd, poly =  u.generate_dataset(K, x_vol, y_vol, ks)
        # fig, _ = u.plot_CSD(x, y, csd, poly)    
     
        # u.save_datapoints(K, C_DD, C_DG, ks, x_vol_range, y_vol_range, cuts, fig)
        try:
            C_DD, C_DG, ks, cuts, x, y, csd, poly =  u.generate_dataset(K, x_vol, y_vol, ks)
        except Exception as e:
            # u.clean_batch() # TODO: Figure this out
            print(f"Execution failed!")
            print(f"Error: {e}")
        else:
            suc+=1
            print(f"Succesfully generated datapoints: {suc}/{N} ({i+1}/{N}).\n\n")
            fig, _ = u.plot_CSD(x, y, csd, poly)    

            u.save_datapoints(K, C_DD, C_DG, ks, x_vol_range, y_vol_range, cuts, fig)
        # try:
        #     sys.stdout.write(f"Succesfully generated datapoints: {suc}/{N} ({i}/{N}).\n\n")
        #     s = time.time()
        #     u.generate_and_save_datapoints(K, x_vol, y_vol)
        #     e = time.time()
        #     suc+=1
        #     sys.stdout.write(f"Last succesful generation took: {(e-s):.3f}[s].")
        #     time.sleep(1)
        #     sys.stdout.flush()
        # except Exception as e:
        #     sys.stdout.write(f"\n\r Execution failed: {e}", end='\r', flush=True)
        #     sys.stdout.flush()
        
        

    main_end = time.time()
    print(f"\nTotal time: {np.abs(main_start-main_end):.3f}[s] -> {np.abs(main_start-main_end)/N:.3f}[s] per datapoint.")    
    sys.stdout.write(f"Succesfully generated datapoints: {suc}/{N}.\n\n")
    # dict = u.load_hfd5()
    # print(dict.keys())

if __name__ == "__main__":
    main()
