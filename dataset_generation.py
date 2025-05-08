import argparse
import sys, time
import numpy as np
import multiprocessing as mp

import utilities.config as c
import utilities.utils as u

def main():
    x_vol = np.linspace(-0.01, 0.05, c.RESOLUTION)
    y_vol = np.linspace(-0.01, 0.05, c.RESOLUTION)

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

    parser.add_argument('--device', type=u.parse_array, default=None,
                        help='The device array in string format. Example: "[[1,1],[1,1]]"')
    
    parser.add_argument('--sensors_radius', type=u.parse_array, default=None,
                        help='The sensors radius in string format. Example: "[1,1,1]"')
    
    parser.add_argument('--sensors_angle', type=u.parse_array, default=None,
                        help='The sensors angle in string format. Example: "[0,0,0]"')
    
    parser.add_argument('--const_sensors_radius', action='store_true', 
                        help='If True, the sensors radius will be constant. Default value is False.')
    
    parser.add_argument('--system_name', type=str, default=None,
                        help='Name of the predefined system configuration (e.g., "sys_2_1", "sys_3_2")')
    
    parser.add_argument('--noise_level', type=str, default=None,
                        help='Prefix for the dataset folder name showcasing the noise level. Remeber the noise amplitdes has to be manually changed in the config.py file (e.g., "n2e5" for noise amplitude 2e-5.)')

    parser.add_argument('--all_euclidean_cuts', action='store_true', 
                        help='If True, all euclidean cuts will be used to generate datapoints. Default value is False.')
    
    parser.add_argument('--cut', type=u.parse_array, default=None,
                        help='The cut in string format. Example: "[[1,0,0],[0,1,0]]"')
    
    args = parser.parse_args()
    N_batch = args.N
    R = args.R
    K = args.K
    S = args.S
    N_dots = K-S if S>0 else K
    device = args.device if args.device is not None else np.ones((1, K-S), dtype=int)
    sensors_radius = args.sensors_radius
    sensors_angle = args.sensors_angle
    system_name = args.system_name
    noise_level = args.noise_level
    all_euclidean_cuts = args.all_euclidean_cuts
    const_sensors_radius = args.const_sensors_radius
    cut = args.cut

    if system_name == "sys_2_1":
        device = np.ones((1,2))
        sensors_angle = [0]
        sensors_radius = [1.5*c.d_DD]
        K = 3
        S = 1
        N_dots = len(u.get_dots_indices(device))
        all_euclidean_cuts = True
    elif system_name == "sys_3_2":
        device = np.array([[0,0,1],[0,1,0],[1,0,0]])
        N_y, N_x = device.shape
        sensors_angle = [0, 3/2*np.pi]
        sensors_radius = [(N_x-0.5)*c.d_DD, (N_y-0.5)*c.d_DD]
        K = 5
        S = 2
        N_dots = len(u.get_dots_indices(device))
        all_euclidean_cuts = True
    elif system_name is not None:
        raise ValueError(f"System name {system_name} not recognized. Not defined in dataset_generation.py.")

    # TODO: check and sanitize the flags a nd input to the generate_dataset function
    # if args.S>1 and args.device is None:
    #     raise ValueError("The device and the number of sensors must be provided when noise is used.")
    # else:
    #     device = np.array(args.device)
    #     S = args.S
    #     N_dots = len(u.get_dots_indices(device))
    #     K = N_dots + S

    # if sensors_radius is not None or sensors_angle is not None:
    #     assert S>0, "The number of sensors must be provided when sensors_radius or sensors_angle are used."
    # if sensors_radius is not None:
    #     assert len(sensors_radius) == S, "The number of sensors radius must be equal to the number of sensors."
    # if sensors_angle is not None:
    #     assert len(sensors_angle) == S, "The number of sensors angle must be equal to the number of sensors."
 
    config_tuple = (K, N_dots, S)
    c.validate_state(*config_tuple)

    print("Dataset generation configuration:")
    print(f"K: {K}, N: {N_dots}, S: {S}")
    print(f"System name: {system_name}")
    print(f"Noise level (folder prefix): {noise_level}")
    print(f"Sensors radius: {sensors_radius if sensors_radius is not None else 'None'}")
    print(f"Sensors angle: {sensors_angle if sensors_angle is not None else 'None'}")
    print(f"All euclidean cuts: {all_euclidean_cuts}")
    print(f"Device:\n {device if device is not None else 'None'}\n\n")
    
    
    for r in range(R):
        print(f"Batch number: {r+1}/{R}.")
        main_start = time.time()
        u.create_paths(config_tuple, system_name=noise_level)
        
        # Prepare arguments for multiprocessing
        pool_args = [(x_vol, y_vol, ks, device, i, N_batch, config_tuple, sensors_radius,
                      sensors_angle, const_sensors_radius, all_euclidean_cuts, cut) for i in range(N_batch)]
        
        # Create a multiprocessing pool
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(u.generate_datapoint, pool_args)
        
        suc = 0
        for i, result in enumerate(results):
            if result is not None:
                suc += 1
            
                C_DD, C_DG, ks, cuts, x_vol, y_vol, csd, poly, sensor, figs, gradients, device, sensors_coordinates = result
                u.save_datapoints(config_tuple, C_DD, C_DG, ks, x_vol, y_vol, cuts, poly, csd,
                                  sensor, figs, gradients, device, sensors_coordinates)
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
