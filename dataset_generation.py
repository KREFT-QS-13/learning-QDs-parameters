import argparse
import json
import sys, time
import numpy as np
import multiprocessing as mp

import src.utilities.config as c
import src.utilities.utils as u

def main():
    x_vol = np.linspace(-0.01, 0.05, c.RESOLUTION)
    y_vol = np.linspace(-0.01, 0.05, c.RESOLUTION)

    parser = argparse.ArgumentParser(description="Generating a dataset.")
    parser.add_argument('-c', type=str, default='configs/data_generation/test.json',
                        help=(
                            "The path to the configuration JSON file containing data generation parameters. Components of the JSON file:\n"
                            "  - x_vol: List, volume range on the X axis (e.g., [-0.01, 0.05])\n"
                            "  - y_vol: List, volume range on the Y axis\n"
                            "  - number_of_data_batches: Integer, number of batches to generate\n"
                            "  - number_of_realizations: Integer, datapoints per batch\n"
                            "  - num_qdots: Integer, number of 'working' quantum dots\n"
                            "  - num_sensors: Integer, number of sensors\n"
                            "  - noise_level: Float, magnitude of noise to apply\n"
                            "  - device_structure: 2D List, binary array encoding device connectivity\n"
                            "  - sensors_radius: List, radii of each sensor\n"
                            "  - sensors_angle: List, angles of each sensor\n"
                            "  - const_sensors_radius: Bool, whether to use constant sensor radii\n"
                            "  - all_euclidean_cuts: Bool, whether to generate all euclidean cuts\n"
                            "  - cut: List or null, specific cut to use (optional)\n"
                            "  - system_name: String, label for this device/system configuration\n"
                            "Default value: configs/data_generation/test.json"
                        ))


    args = parser.parse_args()
    config = json.load(open(args.c))

    N = config["CSD_generation"]["num_qdots"] # number of 'working' quantum dots
    S = config["CSD_generation"]["num_sensors"] # number of sensors
    K = N + S # total number of quantum dots
    config_tuple = (K, N, S)
    c.validate_state(*config_tuple)
    ks = config["CSD_generation"]["ks"]
    
    device = np.array(config["CSD_generation"]["device_structure"])
    sensors_radius = config["CSD_generation"]["sensors_radius"]
    sensors_angle = config["CSD_generation"]["sensors_angle"]
    const_sensors_radius = config["CSD_generation"]["const_sensors_radius"]
    all_euclidean_cuts = config["CSD_generation"]["all_euclidean_cuts"]
    cut = config["CSD_generation"]["cut"]
    system_name = config["CSD_generation"].get("system_name", f'K-{K}_S-{S}')
    noise_level = config["CSD_generation"].get("noise_level", 0.0)
    save_png_images = config["CSD_generation"].get("save_png_images", False)

    assert len(sensors_radius) == S, "The number of sensors radius must be equal to the number of sensors."
    assert len(sensors_angle) == len(sensors_radius), "The number of sensors angle must be equal to the number of sensors."

    R = config["CSD_generation"]["number_of_data_batches"]
    N_batch = config["CSD_generation"]["number_of_realizations"]

    print("Dataset generation configuration:")
    print(f"K: {K}, N: {N}, S: {S}")
    print(f"System name: {system_name}")
    print(f"Noise level (folder prefix): {noise_level}")
    print(f"Sensors radius: {sensors_radius if sensors_radius is not None else 'None'}")
    print(f"Sensors angle: {sensors_angle if sensors_angle is not None else 'None'}")
    print(f"All euclidean cuts: {all_euclidean_cuts}")
    print(f"Device:\n {device if device is not None else 'None'}\n\n")
    print(f"Save PNG images: {save_png_images}")
    
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
