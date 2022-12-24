import numpy as np
from spde import spde
from tqdm import tqdm



def runsim(argv):
    modstr = ["Stationary Isotropic", "Stationary Anistropic", "Non-stationary Simple Anisotropic","Non-stationary Complex Anisotropic"]
    print("Simulating from " + modstr[argv-1] + "...")
    mod = spde(model = argv)
    mod.load()
    res = np.zeros(100)
    for i in tqdm(range(100)):
        res[i] = mod.sim(verbose = False)


def main():
    runsim(1)
    runsim(2)
    runsim(4)

if __name__ == "__main__":
   main()