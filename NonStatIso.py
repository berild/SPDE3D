import numpy as np
from scipy import sparse
from ah3d2 import AH
from sksparse.cholmod import cholesky
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
importr("Matrix")
import os
from grid import Grid
robj.r.source("rqinv.R")
from scipy.optimize import minimize

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

def rqinv(Q):
    tshape = Q.shape
    Q = Q.tocoo()
    r = Q.row
    c = Q.col
    v = Q.data
    tmpQinv =  np.array(robj.r.rqinv(robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))
    return(sparse.csc_matrix((np.array(tmpQinv[:,2],dtype = "float32"), (np.array(tmpQinv[:,0],dtype="int32"), np.array(tmpQinv[:,1],dtype="int32"))), shape=tshape))

class NonStatIso:
    #mod3: kappa(0:27), gammaXY (27),gammaZ (28), sigma(29)
    def __init__(self,grid=None,par=None):
        assert par is None or (par.size==30 if par is not None else False)
        self.grid = Grid() if grid is None else grid
        self.n = self.grid.M*self.grid.N*self.grid.P
        self.kappa = np.array([0.5]*27) if par is None else par[0:27]
        self.gammaXY = 0.5 if par is None else par[27]
        self.gammaZ = 0.5 if par is None else par[28]
        self.tau = 0.5 if par is None else par[29]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        self.V = self.grid.hx*self.grid.hy*self.grid.hz
        self.Dv = self.V*sparse.eye(self.grid.M*self.grid.N*self.grid.P)
        self.iDv = 1/self.V*sparse.eye(self.grid.M*self.grid.N*self.grid.P)
        self.grid.basisH()
        self.grid.basisN()
        self.Q = None
        self.Q_fac = None
        self.mvar = None
        self.data = None
        self.r = None
        self.S = None
        self.opt_steps = 0
        self.verbose = False
        self.likediff = 1000
        self.grad = True
        self.like = -10000
        self.jac = np.array([-100]*30)
        self.loaded = False
        self.truth = None

    def setGrid(self,grid):
        self.grid = grid
        self.n = self.grid.M*self.grid.N*self.grid.P
        self.V = self.grid.hx*self.grid.hy*self.grid.hz
        self.Dv = self.V*sparse.eye(self.grid.M*self.grid.N*self.grid.P)
        self.iDv = 1/self.V*sparse.eye(self.grid.M*self.grid.N*self.grid.P)
        self.grid.basisH()
        self.grid.basisN()

    
    def getPars(self):
        return(np.hstack([self.kappa,self.gammaXY,self.gammaZ,self.tau]))

    def load(self,simple = False):
        simmod = np.load("./simulation_study_models/NI.npz")
        self.kappa = simmod['kappa']*1
        self.gammaXY = simmod['gammaXY']*1
        self.gammaZ = simmod['gammaZ']*1
        self.sigma = simmod['sigma']*1
        self.tau = np.log(1/np.exp(self.sigma)**2)
        Hs = np.diag([np.exp(self.gammaXY),np.exp(self.gammaXY),np.exp(self.gammaZ)]) + np.zeros((self.n,6,3,3))
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa)))
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        self.Q = A_mat.transpose()@self.iDv@A_mat
        if not simple:  
            self.Q_fac = self.cholesky(self.Q)
            assert(self.Q_fac != -1)

    def fit(self,data, r, S = None,verbose = False, fgrad = True, par = None):
        #mod3: kappa(0:27), gammaXY (27),  gammaZ (28), sigma(29)
        if par is None:
            par = np.hstack([[-0.5]*27,2,-1,4])
        self.data = data
        self.r = r
        self.S = S
        self.opt_steps = 0
        self.grad = fgrad
        self.verbose = verbose
        if self.grad:
            res = minimize(self.logLike, x0 = par,jac = True, method = "BFGS",tol = 1e-4)
            res = res['x']
        else:    
            res = minimize(self.logLike, x0 = par, tol = 1e-4)
            res = res['x']
        self.kappa = res[0:27]
        self.gammaXY = res[27]
        self.gammaZ = res[28]
        self.tau = res[29]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        return(res)

    def fitTo(self,simmod,dho,r,num,verbose = False, fgrad = True, par = None):
        if simmod == 3:
            tmp = np.load("./simulation_study_models/NI.npz")
            self.truth = np.hstack([tmp['kappa'],tmp['gammaXY'],tmp['gammaZ'],np.log(1/np.exp(tmp['sigma'])**2)])
        if par is None:
            par = np.load('./simulation_study_models/initNI.npy')
        mods = np.array(['SI','SA','NI','NA'])
        dhos = np.array(['100','10000','27000'])
        rs = np.array([1,10,100])
        tmp = np.load('./simulations/' + mods[simmod-1] + '-'+str(num)+".npz")
        self.data = (tmp['data']*1)[tmp['locs'+dhos[dho-1]],:(rs[r-1])]
        self.r = rs[r-1]
        self.S = np.zeros((self.n))
        self.S[tmp['locs'+dhos[dho-1]]*1] = 1
        self.S = sparse.diags(self.S)
        self.S =  delete_rows_csr(self.S.tocsr(),np.where(self.S.diagonal() == 0))
        res = self.fit(data = self.data, r=self.r, S = self.S,verbose = verbose, fgrad = fgrad,par = par)
        np.savez('./fits/' + mods[simmod-1] + '-NI-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz', par = res)
        return(True)

    def loadFit(self, simmod=None, dho=None, r=None, num=None, file = None):
        if file is None:
            mods = np.array(['SI','SA','NI','NA'])
            dhos = np.array(['100','1000','10000'])
            rs = np.array([1,10,100])
            file = './fits/' + mods[simmod-1] + '-NI-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz'
            print(file)
        fitmod = np.load(file)
        self.S = sparse.eye(self.n)
        par =fitmod['par']*1
        self.kappa = par[0:27]
        self.gammaXY = par[27]
        self.gammaZ = par[28]
        self.tau = par[29]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        Hs = np.diag([np.exp(self.gammaXY),np.exp(self.gammaXY),np.exp(self.gammaZ)]) + np.zeros((self.n,6,3,3))
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa))) 
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = self.cholesky(self.Q)
        self.mvar = rqinv(self.Q).diagonal()

    def sample(self,n = 1):
        z = np.random.normal(size = self.n*n).reshape(self.n,n)
        data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n*n).reshape(self.n,n)*np.exp(self.sigma)
        return(data)

    def sim(self):
        if not self.loaded:
            self.load()
        mods = []
        for file in os.listdir("./simulations/"):
            if file.startswith("NI-"):
                mods.append(int(file.split("-")[1].split(".")[0]))
        if not mods:
            num = 1
        else:
            num = max(mods) + 1 
        self.data = self.sample(n=100)
        np.savez('./simulations/NI-'+ str(num) +'.npz', data = self.data, locs100 = np.random.choice(np.arange(self.n), 100, replace = False), locs10000 = np.random.choice(np.arange(self.n), 10000, replace = False), locs27000 = np.arange(self.n))
        return(True)

    def setQ(self,par = None,S=None,simple = False):
        if par is None:
            assert(self.kappa is not None and self.gammaXY is not None and self.gammaZ is not None and self.sigma is not None)
        else:
            self.kappa = par[0:27]
            self.gammaXY = par[27]
            self.gammaZ = par[28]
            self.tau = par[29]
            self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        if S is not None:
            self.S = S
        Hs = np.diag([np.exp(self.gammaXY),np.exp(self.gammaXY),np.exp(self.gammaZ)]) + np.zeros((self.n,6,3,3))
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa))) 
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = self.cholesky(self.Q)
        if not simple:
            self.Q_fac = self.cholesky(self.Q)

    def cholesky(self,Q):
        try: 
            Q_fac = cholesky(Q)
        except:
            print("Supernodal or negative definite precision matrix... continue")
            return(-1)
        else:
            return(Q_fac)

    def simpleMvar(self,Q_fac,Q, Qc_fac = None,n = 100):
        z = np.random.normal(size = self.n*n).reshape(self.n,n)
        Q = Q.tocoo()
        r = Q.row
        c = Q.col
        d = Q.data
        tmp = Q_fac.apply_Pt(Q_fac.solve_Lt(z,use_LDLt_decomposition=False))
        mt = tmp.mean(axis=1)
        res = ((tmp[r,:] - mt[r,np.newaxis])*(tmp[c,:]-mt[c,np.newaxis])).mean(axis=1)
        tot=sparse.csc_matrix((res, (r, c)), shape=(self.n,self.n))
        if Qc_fac is not None:
            tmp2 = Qc_fac.apply_Pt(Qc_fac.solve_Lt(z,use_LDLt_decomposition=False))
            mt2 = tmp2.mean(axis=1)
            res2 = ((tmp2[r,:] - mt2[r,np.newaxis])*(tmp2[c,:]-mt2[c,np.newaxis])).mean(axis=1)
            tot2=sparse.csc_matrix((res2, (r, c)), shape=(self.n,self.n))
            return((tot,tot2))
        else:
            return(tot)

    def logLike(self, par):
        #mod3: kappa(0:27), gammaXY (27), gammaZ (28), sigma(29)
        data  = self.data
        Hs = np.diag([np.exp(par[27]),np.exp(par[27]),np.exp(par[28])]) + np.zeros((self.n,6,3,3))
        lkappa = self.grid.evalB(par = par[0:27])
        Dk =  sparse.diags(np.exp(lkappa)) 
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Q = A_mat.transpose()@self.iDv@A_mat
        Q_c = Q + self.S.transpose()@self.S*np.exp(par[29])
        Q_fac = self.cholesky(Q)
        Q_c_fac= self.cholesky(Q_c)
        if (Q_fac == -1) or (Q_c_fac == -1):
            if self.grad:
                return((self.like,self.jac))
            else:
                return(self.like)
        mu_c = Q_c_fac.solve_A(self.S.transpose()@data*np.exp(par[29]))
        if self.r == 1:
            data = data.reshape(data.shape[0],1)
            mu_c = mu_c.reshape(mu_c.shape[0],1)
        if self.grad:
            if np.abs(self.likediff) < 0.0001:
                Qinv =  rqinv(Q) 
                Qcinv = rqinv(Q_c)
            else:
                Qinv,Qcinv = self.simpleMvar(Q_fac,Q,Q_c_fac)


            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[29]/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c.transpose()@Q@mu_c).diagonal().sum() - np.exp(par[29])/2*((data - self.S@mu_c).transpose()@(data-self.S@mu_c)).diagonal().sum()
            g_par = np.zeros(30)
            g_par[29] = self.S.shape[0]*self.r/2 - 1/2*(Qcinv@self.S.transpose()@self.S*np.exp(par[29])).diagonal().sum()*self.r - 1/2*((data - self.S@mu_c).transpose()@(data - self.S@mu_c)).diagonal().sum()*np.exp(par[29])
            
            Hs_par = np.diag([np.exp(par[27]),np.exp(par[27]),0]) + np.zeros((self.n,6,3,3)) 
            A_par = -AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
            Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
            g_par[27] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

            Hs_par = np.diag([0,0,np.exp(par[28])]) + np.zeros((self.n,6,3,3))
            A_par = -AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
            Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
            g_par[28] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

            for i in range(27):
                A_par = self.Dv@sparse.diags(self.grid.bs[:,i]*np.exp(lkappa))
                Q_par = A_par.transpose()@self.iDv@A_mat + A_mat.transpose()@self.iDv@A_par
                g_par[i] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

            like =  -like/(self.S.shape[0]*self.r)
            self.likediff = like - self.like
            self.like = like
            jac =  -g_par/(self.S.shape[0]*self.r)
            self.opt_steps = self.opt_steps + 1
            if self.verbose:
                if self.truth is not None:
                    print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%(np.mean(par[0:27]- self.truth[0:27])), "\u03B3_XY = %2.2f"%(par[27]-self.truth[27]), "\u03B3_Z = %2.2f"%(par[28]-self.truth[28]), "\u03C3 = %2.2f"%(par[29]-self.truth[29]))
                else:
                    print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%(np.mean(par[0:27])), "\u03B3_XY = %2.2f"%par[27], "\u03B3_Z = %2.2f"%par[28], "\u03C3 = %2.2f"%par[29])
            return((like,jac))
        else: 
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[189]/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c.transpose()@Q@mu_c).diagonal().sum() - np.exp(par[189])/2*((data - self.S@mu_c).transpose()@(data-self.S@mu_c)).diagonal().sum()
            like = -like/(self.S.shape[0]*self.r)
            self.like = like
            self.opt_steps = self.opt_steps + 1
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like))
            return(like)