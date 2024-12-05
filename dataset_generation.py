import argparse
import sys, time
import numpy as np
import multiprocessing as mp

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

    parser.add_argument('--device', type=u.parse_array, default=np.ones((1,2), dtype=int),
                        help='The device array in string format. Example: "[[1,1],[1,1]]"')
    
    parser.add_argument('--sensors_radius', type=u.parse_array, default=None,
                        help='The sensors radius in string format. Example: "[1,1,1]"')
    
    parser.add_argument('--sensors_angle', type=u.parse_array, default=None,
                        help='The sensors angle in string format. Example: "[0,0,0]"')
    
    parser.add_argument('--system_name', type=str, default=None)


    args = parser.parse_args()
    N_batch = args.N
    R = args.R
    K = args.K
    sensors_radius = args.sensors_radius
    sensors_angle = args.sensors_angle
    system_name = args.system_name

    if args.S>1 and args.device is None:
        raise ValueError("The device and the number of sensors must be provided when noise is used.")
    else:
        device = np.array(args.device)
        print(f"Device configuration (shape={device.shape}):\n{device}")
        S = args.S
        N_dots = len(u.get_dots_indices(device))
        K = N_dots + S

    if system_name == "fixed_4_2":
        device = np.ones((2,2))
        sensors_radius = [0,3/2*np.pi]
        K = 6
        S = 2
        N_dots = len(u.get_dots_indices(device))


    if sensors_radius is not None or sensors_angle is not None:
        assert S>0, "The number of sensors must be provided when sensors_radius or sensors_angle are used."
    if sensors_radius is not None:
        assert len(sensors_radius) == S, "The number of sensors radius must be equal to the number of sensors."
    if sensors_angle is not None:
        assert len(sensors_angle) == S, "The number of sensors angle must be equal to the number of sensors."

        
    config_tuple = (K, N_dots, S)
    c.validate_state(*config_tuple)
    
    for r in range(R):
        print(f"Batch number: {r+1}/{R}.")
        main_start = time.time()
        u.create_paths(config_tuple)
        
        # Prepare arguments for multiprocessing
        pool_args = [(x_vol, y_vol, ks, device, i, N_batch, 
                      config_tuple, sensors_radius, sensors_angle) for i in range(N_batch)]
        
        # Create a multiprocessing pool
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(u.generate_datapoint, pool_args)
        
        suc = 0
        for i, result in enumerate(results):
            if result is not None:
                suc += 1
            
                C_DD, C_DG, ks, cut, x_vol, y_vol, fig, fig_gradient, device, sensors = result
                u.save_datapoints(config_tuple, C_DD, C_DG, ks, x_vol_range, y_vol_range, 
                                  cut, fig, fig_gradient, device, sensors)
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
