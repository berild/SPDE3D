import sys, getopt
import numpy as np
from spde import spde
from joblib import Parallel, delayed
from functools import partial
import os


def fit(version,model,data,vers):
    mod = spde(model = model)
    res = mod.fitTo(data,vers[version,1],vers[version,2], vers[version,0],verbose = False)
    return(res)


def fitPar(model,data,start):
    vers = findFits(model,data,start)
    print(str(vers.shape[0])+" of 900")
    fit_ = partial(fit, model=model, data = data, vers=vers)
    res = Parallel(n_jobs=10,verbose = 100)(delayed(fit_)(i) for i in range(vers.shape[0]))
    return(res)


def findFits(model, data, start):
    if start == -1:
        start = 1
        end = 101
    elif (start + 33)>101:
        end = 101
    else:
        end = start+ 33
    if model != data:
        vers = np.array([[i,j,k] for i in range(start,end) for j in range(1,3) for k in range(1,3)])
    else:
        vers = np.array([[i,j,k] for i in range(start,end) for j in range(1,4) for k in range(1,4)])
    modstr = np.array(["SI", "SA", "NI","NA"])
    dho = np.array(["100","10000","27000"])
    r = np.array(["1","10","100"])
    for file in os.listdir("./fits/"):
        if file.startswith(modstr[model-1]+"-"+modstr[data-1]):
            tmp = file.split('-')[2:]
            tdho = np.where(dho == tmp[0][3:])[0][0] + 1
            tnum = int(tmp[2].split(".")[0])
            tr = np.where(r==tmp[1][1:])[0][0]+1
            tar = np.array([tnum,tdho,tr]) # num , dho , r
            vers = np.delete(vers,np.where((vers == tar).all(axis=1))[0],axis=0)
    return(vers)

def main(argv):
    modstr = ["Stationary Isotropic", "Stationary Anistropic", "Non-stationary Simple Anisotropic","Non-stationary Complex Anisotropic","Stationary Anistropic new"]
    mods = None
    start = int(argv[1])
    mods = np.array([argv[0],argv[0]])
    res = fitPar(int(mods[0]),int(mods[1]),start)
    return(res)

if __name__ == "__main__":
   main(sys.argv[1:])