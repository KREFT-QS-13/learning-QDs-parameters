import numpy as np
import argparse

import sys
sys.path.append('./qdarts')

def generate_capacities_matrices(K, mean, std):
    pass

def generate_dataset(K, x_vol, y_vol, mean, std):
    C_DD, C_Dg = generate_capacities_matrices(K, mean, std)
    pass

def save_datapoints():
    pass

def generate_cuts(K):
    pass

def main():
    mean = 5.0 #
    std = 0.05 #
    x_vol =  np.linspace(0, 0.05, 500)
    y_vol =  np.linspace(0, 0.05, 500)
    parser = argparse.ArgumentParser(description="Generating a dataset")

    parser.add_argument('--K', type=np.int64, nargs=1, default=2, help='Set the number of quantum dots in the system. Default vaule 2.')
    #TODO: Later add noise parameter
    # parser.add_argument('--NOISE', type=n, nargs=1, help='')

    args = parser.parse_args()

    generate_dataset(args.K, x_vol, y_vol, mean, std)

    
if __name__ == "__main__":
    main()
