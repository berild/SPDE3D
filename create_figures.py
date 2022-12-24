import sys
import numpy as np
from mission import mission
from spde import spde
from tqdm import tqdm
import matplotlib.pyplot as plt



def main(argv):
    if argv[0] == "4":
        # simulation study non-stat model
        mod = spde(model=4)
        mod.load()
        mod.plot(version = "mvar")
        mod.plot(version = "mcorr", pos= [26,26,20])
    elif argv[0] == "6":
        # SINMOD observations
        mis = mission(model = 4,pars = True)
        mis.plot(version=4, tidx=int(10.5*6))
        mis.plot(version=4, tidx=int(11.5*6))
        mis.plot(version=4, tidx=int(12.5*6))
        mis.plot(version=4, tidx=int(13.5*6))
        mis.plot(version=4, tidx=int(14.5*6))
    elif argv[0] == "7":
        # prior field non-stat
        mis = mission(model = 4,pars = True)
        mis.plot(version=1)
        mis.plot(version=2)
        mis.plot(version=3, pos = [22,10,0])
    elif argv[0] == "8":
        # measurments figure
        mis = mission(model = 4,pars = True)
        mis.init_auv()
        mis.plot_assimilate()
    elif argv[0] == "9":
        # application  results figure
        mis1 = mission(model = 2,pars = True)
        mis1.init_auv()
        mis2 = mission(model = 2,pars = True)
        mis2.init_auv()
        fold = np.arange(9) + 1
        folds = list([np.arange(8450,15910),np.arange(29500,39990), np.arange(45000,65100),np.arange(69000,87950),np.arange(107300,126100),np.arange(129990,149200),
                    np.arange(153150,172050),np.arange(176100,195670),np.arange(222600,235300)])
        pros = np.arange(0,0.99,0.05)
        rep = 10
        err1 = np.zeros((len(fold)*rep,pros.shape[0]+1,2))
        err2 = np.zeros((len(fold)*rep,pros.shape[0]+1,2))
        tot = mis1.mdata.shape[0]
        counts = np.array([sum(mis1.mdata['fold']==i) for i in fold])
        for i in range(len(fold)):
            for l in range(rep):
                rfold = np.append(fold[i],np.random.choice(np.append(fold[(i+1):],fold[:i]),fold.shape[0]-1,replace = False))
                for j in tqdm(range(pros.shape[0])):
                    tsum = 0
                    gpros = round(pros[j]*tot)
                    idxs = np.array([],dtype = 'int32')
                    for k in rfold:
                        tsum = tsum + counts[k-1]
                        if (tsum > gpros):
                            rem = gpros - (tsum - counts[k-1])
                            idxs = np.append(idxs, np.where(mis1.mdata['fold']==k)[0][:rem])
                            break
                        else:
                            idxs = np.append(idxs,np.where(mis1.mdata['fold']==k)[0])
                    if idxs.shape[0] == 0:
                        tmpidx = np.arange(mis1.mdata.shape[0])
                    else:
                        mis1.update(idx = idxs)
                        mis2.update(idx = idxs)
                        tmpidx = np.delete(np.arange(mis1.mdata.shape[0]), idxs)
                    tmp1 = mis1.predict(idx = tmpidx)
                    tmp2 = mis2.predict(idx = tmpidx)
                    sigma1 = np.sqrt(mis1.auv.var(simple = True))
                    sigma2 = np.sqrt(mis2.auv.var(simple = True))
                    err1[i*rep + l,j,0] = mis1.RMSE(tmp1,mis1.mdata['data'][tmpidx])
                    err2[i*rep + l,j,0] = mis2.RMSE(tmp2,mis2.mdata['data'][tmpidx])
                    err1[i*rep + l,j,1] = mis1.CRPS(tmp1,mis1.mdata['data'][tmpidx],sigma1[mis1.mdata['idx'][tmpidx]])
                    err2[i*rep + l,j,1] = mis2.CRPS(tmp2,mis2.mdata['data'][tmpidx],sigma2[mis2.mdata['idx'][tmpidx]])
                    mis1.auv.loadKeep()
                    mis2.auv.loadKeep()
        merr1 = err1.mean(axis=0)
        serr1 = err1.std(axis=0)
        merr2 = err2.mean(axis=0)
        serr2 = err2.std(axis=0)
        fig, ax = plt.subplots(figsize = (12,15),nrows = 2,sharex = True)
        ax[0].set_ylabel('RMSE',fontsize = 18)
        (line1,cap1,_) = ax[0].errorbar(pros[1:]-0.005,merr1[1:-1,0],yerr = serr1[1:-1,0],color = "#1f77b4",fmt='s',ecolor='#1f77b4',capsize=10,linewidth = 3,markersize = 7)
        (line2,cap2,_) = ax[0].errorbar(pros[1:]+0.005,merr2[1:-1,0],yerr = serr2[1:-1,0],color = "#ff7f0e",fmt='s',ecolor='#ff7f0e',capsize=10,linewidth = 3,markersize = 7)
        for cap in cap1:
            cap.set_markeredgewidth(3)
        for cap in cap2:
            cap.set_markeredgewidth(3)
            
        ax[1].set_ylabel('CRPS',fontsize = 18)
        (line1,cap1,_) = ax[1].errorbar(pros[1:]-0.005,merr1[1:-1,1],yerr = serr1[1:-1,1],color = "#1f77b4",fmt='s',ecolor='#1f77b4',capsize=10,linewidth = 3,markersize = 7)
        (line2,cap2,_) = ax[1].errorbar(pros[1:]+0.005,merr2[1:-1,1],yerr = serr2[1:-1,1],color = "#ff7f0e",fmt='s',ecolor='#ff7f0e',capsize=10,linewidth = 3,markersize = 7)
        for cap in cap1:
            cap.set_markeredgewidth(3)
        for cap in cap2:
            cap.set_markeredgewidth(3)
        ax[1].legend([line2,line1], ("Stationary Anisotropic","Non-stationary Anisotropic"),fontsize = 17,fancybox=True, shadow=True,bbox_to_anchor=(0.86, -0.05),  ncol = 2,markerscale=2)
        ax[0].set_xticks(np.arange(0.05,0.99,0.05), labels = ['%d'%x+"%" for x in np.arange(5,99,5)],fontsize=15)
        ax[1].set_xticks(np.arange(0.05,0.99,0.05), labels = ['%d'%x+"%" for x in np.arange(5,99,5)],fontsize=15)
        fig.tight_layout()
        ax[0].grid()
        ax[1].grid()
        ax[0].tick_params(axis='y', which='major', labelsize=14)
        ax[1].tick_params(axis='y', which='major', labelsize=14)
        fig.savefig('./figures/appres.png')
        print("Figure saved in './figures/appres.png'")


if __name__ == "__main__":
   main(sys.argv[1:])