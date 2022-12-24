import sys, getopt
import numpy as np
import os
from spde import spde

def print1(res):
    lines = list(["\u03BA      ","\u03B3      ","\u03C4      "])
    for j in range(3):
        lines.append("")
        for i in range(9):
            lines[j] = lines[j] + "| %.5f "%res[1][j,i]
    print("Stationary Isotropic model \n")
    print("DHO    |           100          |         10000          |         27000 \n")
    print("Real.  |   1   |   10   |  100  |   1   |   10   |  100  |   1   |   10   |   100\n")
    print(lines[0])
    print(lines[1])
    print(lines[2])

def print2(res):
    lines = list(["\u03BA      ","\u03B3      ","vx      ","vy      ","vz     ","\u03C1_1      ","\u03C1_2      ","\u03C4      "])
    for j in range(8):
        lines.append("")
        for i in range(9):
            lines[j] = lines[j] + "| %5.4f"%res[0][j,i] + "(%.5f) "%res[1][j,i]
    print("Stationary Anisotropic model \n")
    print("DHO    |           100          |         10000          |         27000  \n")
    print("Real.  |   1   |   10   |  100  |   1   |   10   |  100  |   1   |   10   |   100\n")
    for j in range(8):
        print(lines[j])

def print3(res):
    lines = list(["\u03BA      ","\u03B3_1      ","\u03B3_2      ","\u03B3_3      ","\u03C1_1      ","\u03C1_2      ","\u03C1_3      ","\u03C4      "])
    for j in range(8):
        lines.append("")
        for i in range(9):
            lines[j] = lines[j] + "| %.5f "%res[1][j,i]
    print("Non-stationary Isotropic model \n")
    print("DHO    |                       100                      |                     10000                      |                     27000             \n")
    print("Real.  |       1       |       10       |      100      |       1       |       10       |      100      |       1       |       10       |       100    \n")
    for j in range(8):
        print(lines[j])
        

def print4(res):
    lines = list(["\u03BA       ","\u03B3      ","vx      ","vy      ","vz     ","\u03C1_1      ","\u03C1_2      ","\u03C4       "])
    for j in range(8):
        lines.append("")
        for i in range(9):
            lines[j] = lines[j] + "| %.5f "%res[1][j,i]
    print("Non-stationary Anisotropic model \n")
    print("DHO      |             100              |            10000             |           27000              \n")
    print("Real.    |    1    |    10    |   100   |    1    |    10    |   100   |    1    |    10    |    100  \n")
    for j in range(8):
        print(lines[j])


def runres(argv):
    vers = np.array([[i,j,k] for i in range(1,101) for j in range(1,4) for k in range(1,4)])
    modstr = np.array(["SI", "SA", "NI", "NA"])
    dho = np.array(["100","10000","27000"])
    r = np.array(["1","10","100"])
    model = argv
    count = 0
    npars = 0
    for file in os.listdir("./fits/"):
        if file.startswith(modstr[model-1]+"-"+modstr[model-1]):
            count = count + 1
            if count == 1:
                npars = (np.load("./fits/"+file)['par']*1).shape[0]
    pars = np.zeros((count,npars + 2))
    count = 0
    for file in os.listdir("./fits/"):
        if file.startswith(modstr[model-1]+"-"+modstr[model-1]):
            par = (np.load("./fits/"+file)['par']*1)
            tmp = file.split('-')[2:]
            tdho = np.where(dho == tmp[0][3:])[0][0] + 1
            tr = np.where(r==tmp[1][1:])[0][0] + 1
            pars[count,:] = np.hstack([par,tdho,tr])
            count = count + 1
    #np.savez(file = modstr[model-1]+"-"+modstr[model-1] + "-pars",pars = pars)
    res = list([np.zeros((npars,9)),np.zeros((npars,9))])
    mod = spde(model = model)
    mod.load(simple = True)
    truth = mod.getPars()
    for i in range(3): #dho
        for j in range(3): #r
            if model == 4:
                res[0][:54,(i)*3 + j] = pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,:54].mean(axis=0) - truth[:54]
                res[1][:54,(i)*3 + j] = np.sqrt(np.mean((pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,:54] - truth[:54][np.newaxis,:])**2,axis = 0))
                res[0][54:189,(i)*3 + j] = np.abs(pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,54:189]).mean(axis=0)- np.abs(truth[54:189])
                res[1][54:189,(i)*3 + j] = np.sqrt(np.mean((np.abs(pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,54:189]) - np.abs(truth[54:189])[np.newaxis,:])**2,axis = 0))
                res[0][189,(i)*3 + j] = pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,189].mean(axis=0) - truth[189]
                res[1][189,(i)*3 + j] = np.sqrt(np.mean((pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,189] - truth[189])**2,axis = 0))
            elif model == 2:
                res[0][:2,(i)*3 + j] = pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,:2].mean(axis=0) - truth[:2]
                res[1][:2,(i)*3 + j] = np.sqrt(np.mean((pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,:2] - truth[:2][np.newaxis,:])**2,axis = 0))
                res[0][2:7,(i)*3 + j] = np.abs(pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,2:7]).mean(axis=0)- np.abs(truth[2:7])
                res[1][2:7,(i)*3 + j] = np.sqrt(np.mean((np.abs(pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,2:7]) - np.abs(truth[2:7])[np.newaxis,:])**2,axis = 0))
                res[0][7:,(i)*3 + j] = pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,7:].mean(axis=0) - truth[7:]
                res[1][7:,(i)*3 + j] = np.sqrt(np.mean((pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,7] - truth[7])**2))
            else: 
                res[0][:,(i)*3 + j] = pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,:].mean(axis=0) - truth
                res[1][:,(i)*3 + j] = np.sqrt(np.mean((pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars][0,:,:] - truth)**2,axis = 0))
    #np.savez(file = modstr[model-1]+"-"+modstr[model-1] + "-results",res = res)
    if model == 1:
        print1(res)
    elif model == 2:
        print2(res)
    elif model == 3:
        print3(res)
    elif model == 4:
        res2 = list([np.zeros((8,9)),np.zeros((8,9))])
        for i in range(7):
            res2[0][i,:] = res[0][i*27:(i+1)*27,:].mean(axis=0)
            res2[1][i,:] = res[1][i*27:(i+1)*27,:].mean(axis=0)
        res2[0][7,:] = res[0][189,:]
        res2[1][7,:] = res[1][189,:]
        print4(res2)
    return(True)

def main():
    runres(1)
    runres(2)
    runres(4)

if __name__ == "__main__":
   main()