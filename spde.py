import sys
import numpy as np
from scipy import sparse
from grid import Grid
import plotly.graph_objs as go
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
importr("Matrix")
robj.r.source("rqinv.R")

def is_int(val):
    try:
        _ = int(val)
    except ValueError:
        return(False)
    return(True)

def rqinv(Q):
    tshape = Q.shape
    Q = Q.tocoo()
    r = Q.row
    c = Q.col
    v = Q.data
    tmpQinv =  np.array(robj.r.rqinv(robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))
    return(sparse.csc_matrix((np.array(tmpQinv[:,2],dtype = "float32"), (np.array(tmpQinv[:,0],dtype="int32"), np.array(tmpQinv[:,1],dtype="int32"))), shape=tshape))

class spde:
    '''
    Non-Stationary Anisotropic model in 2D and 3D. The Gaussian Random Field is approximated with a Stochastic Partial Differential Equation
    with the matérn covariance function as stationary solution. 
    

    Parameters
    ----------
    grid : class
        Self defined grid from the grid class. If not specified the grid will be default. 
    par : array
        Weights of the basis splines for each parameter. Must have length 136.
    model: int
        Defines the type of model (1 = stationary isotropic, 2 = stationary bidirectional anistropy, 
        3 = non-stationary unidirectional anisotropy, 4 = non-stationary biderectional anisotropy)

    Attributes
    ----------
    define : str
        This is where we store arg,
    '''
    def __init__(self, model = None, par = None):
        self.grid = Grid()
        self.model = model
        if self.model is not None:
            self.define(model = self.model,par=par)
        else:
            self.mod = None 

    def setGrid(self,M=None,N = None, P = None, x = None, y = None, z = None):
        self.grid.setGrid(M = M, N = N, P = P, x = x, y = y, z = z)
        self.mod.setGrid(self.grid)

    def setQ(self,par = None,S = None,simple = False):
        self.mod.setQ(par=par,S=S)

    # fix par for models
    def define(self, model = None, par = None):
        assert(model is not None or self.model is not None)
        if model is not None:
            self.model = model 
        if (self.model==1):
            from StatIso import StatIso
            self.mod = StatIso(grid = self.grid, par=par)
        elif (self.model==2):
            from StatAnIso import StatAnIso
            self.mod = StatAnIso(grid = self.grid, par=par)
        elif (self.model==3):
            from NonStatIso import NonStatIso
            self.mod = NonStatIso(grid = self.grid, par=par)
        elif (self.model==4):
            from NonStatAnIso import NonStatAnIso
            self.mod = NonStatAnIso(grid = self.grid,par=par)
        elif (self.model==5):
            from ocean import NonStatAnIso
            self.mod = NonStatAnIso(grid = self.grid,par=par)
        else:
            print("Not a implemented model (1-4)...")

    def load(self,model=None,simple = False):
        if model is None:
            if self.mod is None:
                print("No model defined...")
            else:
                print("Loading pre-defined model",self.model)
        else:
            self.define(model = model)
        self.mod.load(simple = simple)

    def loadFit(self, simmod=None, dho=None, r=None, num=None, model = None, file = None):
        if model is None:
            if self.mod is None:
                print("No model defined...")
        else:
            self.model = model
            self.define(model = model)
        self.mod.loadFit(simmod = simmod, dho = dho , r = r, num = num , file=file)

    # fix par for models
    def fitTo(self,simmod,dho,r,num,model = None,verbose = False, fgrad = True,par = None):
        if model is None:
            if self.mod is None:
                print("No model defined...")
        else:
            self.model = model
            self.define(model = model)
        success = self.mod.fitTo(simmod, dho, r, num,verbose = verbose, fgrad=fgrad,par = par)
        return(success)
        
    # fix par for models
    def fit(self,data,r, S, par = None,model = None,verbose = False, fgrad = True,end = None):
        if model is None:
            if self.mod is None:
                print("No model defined...")
        else:
            self.model = model
            self.define(model = model)
        res = self.mod.fit(data, r, S, par = par,verbose = verbose, fgrad=fgrad,end = end)
        return(res)

    def sim(self,model=None,verbose=True):
        if model is None:
            if self.mod is None:
                print("No model defined...")
                sys.exit()
            else:
                if verbose:
                    print("Loading pre-defined model ", self.model)
        else:
            self.define(model = model)
        return(self.mod.sim())

    def sample(self, n = 1,model = None, Q_fac = None, sigma = None,simple = False):
        if model is None:
            if self.mod is None:
                print("No model defined...")
        else:
            self.define(model = model)
        #return(self.mod.sample(n = n,Q_fac = Q_fac, sigma = sigma, simple = simple))
        return(self.mod.sample(n = n, simple = simple))
        
    def Mvar(self):
        self.mod.setQ()
        self.mod.mvar = rqinv(self.mod.Q).diagonal()

    def getPars(self):
        assert self.mod is not None
        return(self.mod.getPars())

    
    def Corr(self, pos = None):
        ik = np.zeros(self.mod.n)
        if pos is None:
            pos = np.array([self.grid.M/2,self.grid.N/2, self.grid.P/2])
        k = int(pos[1]*self.grid.M*self.grid.P + pos[0]*self.grid.P + pos[2])
        ik[k] = 1
        cov = self.mod.Q_fac.solve_A(ik)
        if self.mod.mvar is None:
            self.Mvar()
        return(cov/np.sqrt(self.mod.mvar[k]*self.mod.mvar))

    def plot(self,value=None,version="mvar", pos = None):
        if value is None:
            if version == "mvar":
                if self.mod.mvar is None:
                    self.Mvar()
                value = self.mod.mvar
            elif version == "mcorr":
                if self.mod.Q_fac is None:
                    self.mod.setQ()
                if pos is None:
                    pos = np.array([self.grid.M/2,self.grid.N/2, self.grid.P/2])
                value = self.Corr(pos)
            elif version == "real":
                value = self.sample(simple = True)
        fig = go.Figure(go.Volume(
            x=self.grid.sx,
            y=self.grid.sy,
            z= - self.grid.sz,
            value = value,
            opacity=0.4, 
            surface_count=30,
            colorscale='rainbow',
            caps= dict(x_show=False, y_show=False, z_show=False)))
        fig.write_html("test.html",auto_open=True)

