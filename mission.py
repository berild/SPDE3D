from re import S
import numpy as np
import pandas as pd
from spde import spde
import netCDF4
import datetime
from scipy import sparse
from auv import auv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import norm
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.io as pio
from os.path import exists
from tqdm import tqdm


def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = (indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def matchTimestamps(timestamps,time):
    tint =  np.zeros(2)
    ex = False
    tint[1] = len(timestamps)-1
    for i in range(len(timestamps)):
        if timestamps[i] == time[0]:
            tint[0] = i
            ex = True
        if timestamps[i]==time[1]:
            tint[1] = i        
    if ex:
        return(tint.astype('int32'))
    else:
        return(None)

class mission:
    def __init__(self,pars = False, model = 4):
        self.model = 4
        # loading parameters if
        par = None
        if pars:
            par = np.load("./fits/nidelva_27_05_21_model_"+str(model)+ ".npz")['par']*1
        # constructing spde
        self.mod = spde(model = model, par=par)
        # loading SINMOD data
        tmp = np.load("./emulator.npz")
        x = tmp['x']
        y = tmp['y']
        z = tmp['z']
        self.time = tmp['time']
        M = x.shape[0]
        N = y.shape[0]
        P = z.shape[0]
        self.slon = tmp['slon']
        self.slat = tmp['slat']
        self.szll = tmp['szll']
        data = tmp['salinity']
        # setting grid 
        self.mod.setGrid(M,N,P,x,y,z)
        S = sparse.diags((data[:,0]>0)*1)
        self.S = delete_rows_csr(S.tocsr(),np.where(S.diagonal() == 0))
        self.edata = data
        self.mod.setQ(S = self.S)

        # creating independent realization
        # by decorrelating in time
        data = self.S@self.edata
        self.mu = data.mean(axis = 1)
        rho = np.sum((data[:,1:]-self.mu[:,np.newaxis])*(data[:,:(data.shape[1]-1)] - self.mu[:,np.newaxis]),axis = 1)/np.sum((data[:,:(data.shape[1]-1)] - self.mu[:,np.newaxis])**2,axis = 1)
        self.muf = (data[:,1:]-self.mu[:,np.newaxis]) - rho[:,np.newaxis]*(data[:,:(data.shape[1]-1)] - self.mu[:,np.newaxis]) + np.random.normal(0,0.2,data.shape[0]*(data.shape[1]-1)).reshape(data.shape[0],data.shape[1]-1)
        self.rho = rho

    def fit(self,verbose = False):
        print("Starting model fit of model "+ str(self.mod.model))
        if self.model == 4:
            par = np.hstack([np.random.normal(-1,0.5,27),
                    np.random.normal(-1,0.5,27),
                    np.random.normal(1,0.5,27),
                    np.random.normal(1,0.5,27),
                    np.random.normal(0.2,0.2,27),
                    np.random.normal(0.5,0.5,27),
                    np.random.normal(0.5,0.5,27),3])
        else:
            par = None
        self.mod.fit(data=self.muf, par = par,r=self.muf.shape[1],S=self.S,verbose= verbose)
        par = self.mod.getPars()
        np.savez("./fits/nidelva_27_05_21_model_" + str(self.model) +  '.npz', par = par)

    def init_auv(self):
        # loading assimilated mission data
        self.mdata = pd.read_csv("mission.csv",index_col = 0)
        # initializing auv class
        self.auv = auv(self.mod,self.mu,np.sqrt((self.mdata['sd']**2).mean()))

    def update(self, idx=None, fold=None, keep = False):
        assert self.auv_exists, "No AUV defined"
        if idx is not None:
            data = self.mdata.iloc[idx]
            self.auv.update(data = data,keep = keep)
        elif fold is not None:
            if hasattr(fold, "__len__"):
                data = self.mdata[np.array([x in fold for x in self.mdata['fold']])]
                self.auv.update(data = data,keep = keep)
            else:
                data= self.mdata[self.mdata['fold']==fold]
                self.auv.update(data = data,keep = keep)
        else:
            print("No index for update defined")
            return()

    def predict(self,idx = None, fold = None):
        assert self.auv_exists, "No AUV defined"
        if idx is not None:
            return(self.auv.mu[self.mdata['idx'][idx]])
        elif fold is not None:
            if hasattr(fold, "__len__"):
                return(self.auv.mu[self.mdata['idx'][np.array([x in fold for x in self.mdata['fold']])]])
            else:
                return(self.auv.mu[self.mdata['idx'][self.mdata['fold']==fold]])
        else:
            return(self.auv.mu)

    def plot_assimilate(self):
        im = mpimg.imread('./figures/AOOsmall.png')
        tmp = np.load('./figures/AOOdat.npy')
        im_lon = np.linspace(tmp[:,1].min(),tmp[:,1].max(),im.shape[1])
        im_lat = np.linspace(tmp[:,0].min(),tmp[:,0].max(),im.shape[0])
        depth = np.array([0.5,1.5,2.5,3.5,4.5,5.5])
        for i in range(6):
            fig, ax = plt.subplots(figsize =(10,10))
            tmp = self.mdata['idx'][self.mod.grid.sz[self.mdata['idx']]==depth[i]]
            pos_lat = self.slat[tmp]
            pos_lon = self.slon[tmp]
            y = np.zeros(pos_lat.shape)
            x = np.zeros(pos_lon.shape)
            for k in range(pos_lat.size):
                y[k] = np.nanargmin((pos_lat[k]-im_lat)**2)
                x[k] = np.nanargmin((pos_lon[k]-im_lon)**2)
            y = im_lat.shape[0] - y
            ax.imshow(im)
            ax.plot(x,y,'ob',markersize=6)
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig('./figures/folds%.d'%i + '.png', dpi=300)

    def CRPS(self,pred,truth,sigma):
        z = (truth - pred)/sigma
        return(np.mean(sigma*(- 2/np.sqrt(np.pi) + 2*norm.pdf(z) + z*(2*norm.cdf(z)-1))))

    def RMSE(self,pred,truth):
        return(np.sqrt(np.mean((pred-truth)**2)))

    def plot(self,version = 1, filename = None,pos = None,tidx = None):
        
        pio.orca.shutdown_server()
        M = self.mod.grid.M
        N = self.mod.grid.N
        P = self.mod.grid.P
        sx = self.mod.grid.sx
        sy = self.mod.grid.sy
        sz = self.mod.grid.sz
        if version == 1:
            cs = [(0.00, "rgb(127, 238, 240)"),   (0.50, "rgb(127, 238, 240)"),
                (0.50, "rgb(192, 245, 240)"), (0.60, "rgb(192, 245, 240)"),
                (0.60, "rgb(241, 241, 225)"),  (0.70, "rgb(241, 241, 225)"),
                (0.70, "rgb(255, 188, 188)"),  (0.80, "rgb(255, 188, 188)"),
                (0.80, "rgb(245, 111, 136)"),  (1.00, "rgb(245, 111, 136)")]
               
            value = self.S.transpose()@self.mu
            cmin = value.min()+2
            cmax = value.max()+2
            if filename is None:
                filename = "mean"
        elif version == 2: 
            cs = [(0.00, "rgb(245, 245, 245)"),   (0.20, "rgb(245, 245, 245)"),
            (0.20, "rgb(245, 201, 201)"), (0.40, "rgb(245, 201, 201)"),
            (0.40, "rgb(245, 164, 164)"),  (0.60, "rgb(245, 164, 164)"),
            (0.60, "rgb(245, 117, 117)"),  (0.80, "rgb(245, 117, 117)"),
            (0.80, "rgb(245, 67, 67)"),  (1.00, "rgb(245, 67, 67)")]
            self.mod.Mvar()
            value = self.mod.mod.mvar
            cmin = value.min()
            cmax = value.max()
            if filename is None:
                filename = "mvar"
        elif version == 3:
            cs = [(0.00, "rgb(245, 245, 245)"),   (0.20, "rgb(245, 245, 245)"),
                (0.20, "rgb(245, 201, 201)"), (0.40, "rgb(245, 201, 201)"),
                (0.40, "rgb(245, 164, 164)"),  (0.60, "rgb(245, 164, 164)"),
                (0.60, "rgb(245, 117, 117)"),  (0.80, "rgb(245, 117, 117)"),
                (0.80, "rgb(245, 67, 67)"),  (1.00, "rgb(245, 67, 67)")]
            if pos is not None:
                value = self.mod.Corr(pos = pos) 
            else:
                pos = [22,2,0]
                value = self.mod.Corr(pos = [22,2,0])
            cmin = 0
            cmax = 1
            if filename is None:
                filename = "mcorr"

        elif version == 4:
            cs = [(0.00, "rgb(127, 238, 240)"),   (0.50, "rgb(127, 238, 240)"),
                (0.50, "rgb(192, 245, 240)"), (0.60, "rgb(192, 245, 240)"),
                (0.60, "rgb(241, 241, 225)"),  (0.70, "rgb(241, 241, 225)"),
                (0.70, "rgb(255, 188, 188)"),  (0.80, "rgb(255, 188, 188)"),
                (0.80, "rgb(245, 111, 136)"),  (1.00, "rgb(245, 111, 136)")]
            if tidx is not None:
                value = self.edata[:,tidx]
                time = datetime.datetime.fromtimestamp(self.time[tidx]).strftime("%H:%M")
            else:
                value = self.edata[:,0]
                time = datetime.datetime.fromtimestamp(self.time[0]).strftime("%H:%M")
            if filename is None:
                filename = "obs" + time
            cmin = self.mu.min()
            cmax = self.mu.max()

        xarrow = np.array([sx.max()-175,sx.max()-50,sx.max()-50,sx.max()-50,sx.max()-125])
        yarrow = np.array([sy.max()-183,sy.max()-58,sy.max()-133,sy.max()-58,sy.max() -58])
        xdif = self.mod.grid.A/4
        ydif = self.mod.grid.B/4
        fig = go.Figure(data=[go.Isosurface(surface_count=1,z=-sz.reshape(N,M,P)[:,:,0].flatten(), x=sx.reshape(N,M,P)[:,:,0].flatten(), y=sy.reshape(N,M,P)[:,:,0].flatten(),value=value.reshape(N,M,P)[:,:,0].flatten(),isomin = cmin,isomax = cmax,colorscale=cs,colorbar=dict(thickness=20,lenmode = "fraction", len = 0.80, ticklen=10,tickfont=dict(size=30, color='black')))])
        fig.add_trace(go.Isosurface(surface_count=1,z=-sz.reshape(N,M,P)[:,:,1].flatten(), x=sx.reshape(N,M,P)[:,:,1].flatten()+xdif*1, y=sy.reshape(N,M,P)[:,:,1].flatten()-ydif*1,value=value.reshape(N,M,P)[:,:,1].flatten(),isomin = cmin,isomax =cmax,colorscale=cs,showscale = False))
        fig.add_trace(go.Isosurface(surface_count=1,z=-sz.reshape(N,M,P)[:,:,2].flatten(), x=sx.reshape(N,M,P)[:,:,2].flatten()+xdif*2, y=sy.reshape(N,M,P)[:,:,2].flatten()-ydif*2,value=value.reshape(N,M,P)[:,:,2].flatten(),isomin = cmin,isomax =cmax,colorscale=cs,showscale = False))
        fig.add_trace(go.Isosurface(surface_count=1,z=-sz.reshape(N,M,P)[:,:,3].flatten(), x=sx.reshape(N,M,P)[:,:,3].flatten()+xdif*3, y=sy.reshape(N,M,P)[:,:,3].flatten()-ydif*3,value=value.reshape(N,M,P)[:,:,3].flatten(),isomin = cmin,isomax =cmax,colorscale=cs,showscale = False))
        fig.add_trace(go.Isosurface(surface_count=1,z=-sz.reshape(N,M,P)[:,:,4].flatten(), x=sx.reshape(N,M,P)[:,:,4].flatten()+xdif*4, y=sy.reshape(N,M,P)[:,:,4].flatten()-ydif*4,value=value.reshape(N,M,P)[:,:,4].flatten(),isomin = cmin,isomax =cmax,colorscale=cs,showscale = False))
        fig.add_trace(go.Isosurface(surface_count=1,z=-sz.reshape(N,M,P)[:,:,5].flatten(), x=sx.reshape(N,M,P)[:,:,5].flatten()+xdif*5, y=sy.reshape(N,M,P)[:,:,5].flatten()-ydif*5,value=value.reshape(N,M,P)[:,:,5].flatten(),isomin = cmin,isomax =cmax,colorscale=cs,showscale = False))

        if pos is not None:
            fig.add_trace(go.Scatter3d(mode = "markers", x=[sx[pos[1]*45*10 + pos[0]*10 + pos[2]]+xdif*pos[2]], y = [sy[pos[1]*45*10 + pos[0]*10 + pos[2]]-ydif*pos[2]], z=[-sz[pos[1]*45*10 + pos[0]*10 + pos[2]]], marker_symbol = "x", marker_color="midnightblue",marker_size=7,showlegend = False))

        fig.add_trace(go.Scatter3d(x=[0,0,xdif,xdif*2,xdif*3,xdif*4,xdif*5]+sx[0], y=[0,0,-ydif,-ydif*2,-ydif*3,-ydif*4,-ydif*5]+sy[0], z=[0,-0.5,-1.5,-2.5,-3.5,-4.5,-5.5], mode='text',text = ["Depth:","0.5","1.5","2.5","3.5","4.5","5.5"],textfont=dict(family="sans serif",size=25,color="black"),showlegend=False))
        for j in range(6):
            fig.add_trace(go.Scatter3d(x=[0,0]+sx[[0*M*P + 0*P + 0, 0*M*P + (M-1)*P + 0]]+xdif*j, y=[0,0]+sy[[0*M*P + 0*P + 0, 0*M*P + (M-1)*P + 0]]-ydif*j, z=np.array([-0.5,-0.5])-j, mode='lines',line = dict(color='black'),showlegend=False))
            fig.add_trace(go.Scatter3d(x=[0,0]+sx[[0*M*P + 0*P + 0, (N-1)*M*P + 0*P + 0]]+xdif*j, y=[0,0]+sy[[0*M*P + 0*P + 0, (N-1)*M*P + 0*P + 0]]-ydif*j, z=np.array([-0.5,-0.5])-j, mode='lines',line = dict(color='black'),showlegend=False))
            fig.add_trace(go.Scatter3d(x=[0,0]+sx[[(N-1)*M*P + 0*P + 0, (N-1)*M*P + (M-1)*P + 0]]+xdif*j, y=[0,0]+sy[[(N-1)*M*P + 0*P + 0, (N-1)*M*P + (M-1)*P + 0]]-ydif*j, z=np.array([-0.5,-0.5])-j, mode='lines',line = dict(color='black'),showlegend=False))
            fig.add_trace(go.Scatter3d(x=[0,0]+sx[[0*M*P + (M-1)*P + 0, (N-1)*M*P + (M-1)*P + 0]]+xdif*j, y=[0,0]+sy[[0*M*P + (M-1)*P + 0, (N-1)*M*P + (M-1)*P + 0]]+-ydif*j, z=np.array([-0.5,-0.5])-j, mode='lines',line = dict(color='black'),showlegend=False))


        fig.add_trace(go.Scatter3d(x=xarrow, 
                                        y=yarrow,
                                        z=np.array([0,0,0,0,0])-0.5,
                                        line=dict(color='black',width=12),
                                        mode='lines+text',
                                        text=["","", "N","",""],
                                        showlegend=False,
                                        textfont=dict(
                                            family="sans serif",
                                            size=25,
                                            color="LightSeaGreen")))

        camera = dict(
                    eye=dict(x=1.2, 
                            y=-1.2, 
                            z=1.3),
                    center=dict(x=0.2, y=-0.2, z=0.18))
        fig.update_scenes(xaxis_visible=False, 
                            yaxis_visible=False,zaxis_visible=False,camera = camera)
        fig.update_layout(autosize=False,
                        width=650,height=1200,scene_aspectratio=dict(x=1, y=1, z=1.0))
        if version == 4:
            fig.update_layout(title={
                                    'text': time,
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font_size': 30},autosize=False,
                        width=650,height=1200,scene_aspectratio=dict(x=1, y=1, z=1.0))
        else:
            fig.update_layout(autosize=False,
                        width=650,height=1200,scene_aspectratio=dict(x=1, y=1, z=1.0))

        fig.write_html("test.html", auto_open = True)
        if filename is not None:
            fig.write_image("./figures/" + filename + ".png",engine="orca",scale=1)