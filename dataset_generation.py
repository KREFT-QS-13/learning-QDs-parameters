import argparse
import json
import os, time
import numpy as np
import multiprocessing as mp
import glob

import src.utilities.utils as u

def count_umber_of_folders(path: str, prefix: str) -> int:
    return len([f for f in os.listdir(path) if f.startswith(prefix)])

def process_single_config(config_path: str, num_samples: int = 5):
    """
    Process a single config file and generate the specified number of samples.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file
    num_samples : int
        Number of samples to generate for this config (default: 5)
    """
    print(f"\n{'='*80}")
    print(f"Processing config: {config_path}")
    print(f"{'='*80}")
    
    config = json.load(open(config_path))

    Ndots = config["system"]["Ndots"] # number of 'working' quantum dots
    Nsensors = config["system"]["Nsensors"] # number of sensors
    Ngates = config["system"]["Ngates"] # number of gates
    assert Ndots + Nsensors == Ngates, "The number of dots plus the number of sensors must be equal to the number of gates."

    ks = config["CSD_generation"].get("ks", None)
    
    save_png_images = config["CSD_generation"].get("save_png_images", False)
    system_name = config["CSD_generation"].get("system_name", f'N-{Ndots}_S-{Nsensors}')
    path_to_end_folder = config["CSD_generation"].get("path_to_end_folder", "datasets/")
    folder_name =  f'{system_name}__{count_umber_of_folders(path_to_end_folder, system_name)+1}'
    
    Number_of_realizations = config["CSD_generation"]["number_of_realizations"]

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
    print(f"\nGeneration complete for {config_path}!")
    print(f"Saved under the path: {system_folder_path}")
    print(f"Successfully generated: {successful_count} datapoints")
    print(f"Successful/Failed ratio: {successful_count}/{failed_count}")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
    if (successful_count + failed_count) > 0:
        print(f"Average time per datapoint: {(end_time - start_time)/(successful_count + failed_count):.2f} seconds.")
    
    return {
        'config_path': config_path,
        'system_folder_path': system_folder_path,
        'successful_count': successful_count,
        'failed_count': failed_count,
        'time_taken': end_time - start_time
    }

def main():
    parser = argparse.ArgumentParser(description="Generating a dataset.")
    parser.add_argument('-c', type=str, default=None, 
                       help='Path to a single config file. If not provided, processes all JSON files in configs/')
    parser.add_argument('--configs-dir', type=str, default='configs',
                       help='Directory containing config files (default: configs)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples per config. If omitted, uses config["CSD_generation"]["number_of_realizations"] (default there: 5)')

    args = parser.parse_args()
    
    # If a specific config file is provided, process only that one
    if args.c:
        results = [process_single_config(args.c, args.num_samples)]
    else:
        # Process all JSON files in the configs directory
        config_files = glob.glob(os.path.join(args.configs_dir, '*.json'))
        config_files.sort()  # Process in alphabetical order
        
        if not config_files:
            print(f"No JSON config files found in {args.configs_dir}")
            return
        
        print(f"Found {len(config_files)} config files in {args.configs_dir}")
        print(f"Will generate {args.num_samples} samples for each config file")
        
        results = []
        overall_start_time = time.time()
        
        for config_file in config_files:
            result = process_single_config(config_file, args.num_samples)
            results.append(result)
        
        overall_end_time = time.time()
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY - All Configs Processed")
        print(f"{'='*80}")
        total_successful = sum(r['successful_count'] for r in results)
        total_failed = sum(r['failed_count'] for r in results)
        total_time = overall_end_time - overall_start_time
        
        print(f"Total configs processed: {len(results)}")
        print(f"Total successful datapoints: {total_successful}")
        print(f"Total failed datapoints: {total_failed}")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        if (total_successful + total_failed) > 0:
            print(f"Average time per datapoint: {total_time/(total_successful + total_failed):.2f} seconds")
        
        print(f"\nPer-config results:")
        for r in results:
            print(f"  {os.path.basename(r['config_path'])}: {r['successful_count']} successful, {r['failed_count']} failed")

if __name__ == "__main__":
    main()
