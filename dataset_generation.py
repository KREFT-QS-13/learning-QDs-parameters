import argparse
import json
import os, time
import numpy as np
import multiprocessing as mp

import src.utilities.utils as u

def count_umber_of_folders(path: str, prefix: str) -> int:
    return len([f for f in os.listdir(path) if f.startswith(prefix)])

def main():
    parser = argparse.ArgumentParser(description="Generating a dataset.")
    parser.add_argument('-c', type=str, default='configs/data_generation/test.json')

    args = parser.parse_args()
    config = json.load(open(args.c))

    Ndots = config["system"]["Ndots"] # number of 'working' quantum dots
    Nsensors = config["system"]["Nsensors"] # number of sensors
    Ngates = config["system"]["Ngates"] # number of gates
    assert Ndots + Nsensors == Ngates, "The number of dots plus the number of sensors must be equal to the number of gates."

    ks = config["CSD_generation"].get("ks", None)
    
    save_png_images = config["CSD_generation"].get("save_png_images", False)
    system_name = config["CSD_generation"].get("system_name", f'N-{Ndots}_S-{Nsensors}')
    path_to_end_folder = config["CSD_generation"].get("path_to_end_folder", "datasets/")
    folder_name =  f'{system_name}__{count_umber_of_folders(path_to_end_folder, system_name)+1}'
    Number_of_realizations = config["CSD_generation"]["number_of_realizations"] # number of realizations

    # Create the main system folder at the start
    system_folder_path = os.path.join(path_to_end_folder, folder_name)
    os.makedirs(system_folder_path, exist_ok=True)
    print(f"Created system folder: {system_folder_path}")

    print("Dataset generation configuration:")  
    print(f"K: {Ndots + Nsensors}, N: {Ndots}, S: {Nsensors}")
    print(f"System name: {system_name}")
    print(f"Save PNG images: {save_png_images}")
    print(f"Path where the dataset will be saved: {path_to_end_folder}")
    print(f"System folder: {folder_name}")
    print(f"Starting generation of {Number_of_realizations} realizations...")

    start_time = time.time()
    successful_count = 0
    failed_count = 0
    
    for i in range(Number_of_realizations):
        print(f"Generating realization {i+1}/{Number_of_realizations}...")
        success = u.generate_datapoint(config, successful_count, system_folder_path)
        if success:
            successful_count += 1
        else:
            failed_count += 1
            print(f"Failed to generate datapoint. Retrying with next index...")
    
    end_time = time.time()
    print(f"\nGeneration complete!")
    print(f"Saved under the path: {system_folder_path}")
    print(f"Successfully generated: {successful_count} datapoints")
    print(f"Successful/Failed ratio: {successful_count}/{failed_count}")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
    print(f"Average time per datapoint: {(end_time - start_time)/(successful_count + failed_count):.2f} seconds.")

if __name__ == "__main__":
    main()
