import argparse
import utilities.model_utils as mu
from utilities.config import K as config_K, set_global_K

def parse_args():
    parser = argparse.ArgumentParser(description="QDarts Learning Parameters")
    # Add other existing arguments here
    parser.add_argument('--K', type=np.int32, default=config_K, 
                        help='The number of quantum dots in the system. Default vaule 2.')
    return parser.parse_args()






if __name__ == "__main__":
    args = parse_args()
    set_global_K(args.K)
    # Rest of your main code
