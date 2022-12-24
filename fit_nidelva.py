import sys
import numpy as np
from mission import mission
from tqdm import tqdm

def main(argv):
    model = int(argv[0])
    mis = mission(model = model)
    mis.fit(verbose = True)
    return()

if __name__ == "__main__":
   main(sys.argv[1:])