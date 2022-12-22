import numpy as np
from scipy import sparse
from ah3d2 import AH
from sksparse.cholmod import cholesky
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
importr("Matrix")
from scipy.optimize import minimize
import os
from grid import Grid
robj.r.source("rqinv.R")

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


class StatAnIso:
    #mod2: kappa(0), gamma(1), vx(2), vy(3), vz(4), rho1(5), rho2(6), sigma(7)
    def __init__(self,grid=None,par=None):
        assert par is None or (par.size==8 if par is not None else False)
        self.grid = Grid() if grid is None else grid
        self.V = self.grid.hx*self.grid.hy*self.grid.hz
        self.n = self.grid.M*self.grid.N*self.grid.P
        self.kappa = 0.5 if par is None else par[0]
        self.gamma = 0.5 if par is None else par[1]
        self.vx = 0.5 if par is None else par[2]
        self.vy = 0.5 if par is None else par[3]
        self.vz = 0.5 if par is None else par[4]
        self.rho1 = 0.5 if par is None else par[5]
        self.rho2 = 0.5 if par is None else par[6]
        self.tau = 0.5 if par is None else par[7]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        self.Dv = self.V*sparse.eye(self.n)
        self.iDv = 1/self.V*sparse.eye(self.n)
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
        self.jac = np.array([-100]*8)
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
        return(np.hstack([self.kappa,self.gamma,self.vx,self.vy,self.vz,self.rho1,self.rho2,self.tau]))

    def load(self,simple = False):
        simmod = np.load("./simmodels/SA.npz")
        self.kappa = simmod['kappa']*1
        self.gamma = simmod['gamma']*1
        self.vx = simmod['vx']*1
        self.vy = simmod['vy']*1
        self.vz = simmod['vz']*1
        self.rho1 = simmod['rho1']*1
        self.rho2 = simmod['rho2']*1
        self.sigma = simmod['sigma']*1
        self.tau = np.log(1/np.exp(self.sigma)**2)
        Hs = self.getH()
        Dk =  sparse.diags([np.exp(self.kappa)]*(self.n)) 
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        self.Q = A_mat.transpose()@self.iDv@A_mat
        if not simple:
            self.Q_fac = self.cholesky(self.Q)
            assert(self.Q_fac != -1)
            self.loaded = True
        
    def fit(self,data, r, S,par = None,verbose = False, fgrad = True, end = None):
        if par is None:
            par = np.array([-1.5,-1.1,5,4.2,0.04,0.4,0.3,3])
        self.data = data
        self.r = r
        self.S = S
        self.opt_steps = 0
        self.grad = fgrad
        self.verbose = verbose
        self.end = end
        if self.grad:
            res = minimize(self.logLike, x0 = par,jac = True, method = "BFGS",tol = 1e-4)
            res = res['x']
        else:    
            res = minimize(self.logLike, x0 = par, tol = 1e-4)
            res = res['x']
        self.kappa = res[0]
        self.gamma = res[1]
        self.vx = res[2]
        self.vy = res[3]
        self.vz = res[4]
        self.rho1 = res[5]
        self.rho2 = res[6]
        self.tau = res[7]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        return(res)

    def fitTo(self,simmod,dho,r,num,verbose = False, fgrad = True, par = None):
        if simmod == 2:
            tmp = np.load('./simmodels/SA.npz')
            self.truth = np.hstack([tmp['kappa'],tmp['gamma'],tmp['vx'],tmp['vy'],tmp['vz'],tmp['rho1'],tmp['rho2'],np.log(1/np.exp(tmp['sigma'])**2)])
        if par is None:
            par = np.load('./simmodels/initSA.npy')
        mods = np.array(['SI','SA','NI','NA'])
        dhos = np.array(['100','10000','27000'])
        rs = np.array([1,10,100])
        tmp = np.load('./simulations/' + mods[simmod-1] + '-'+str(num)+".npz")
        self.data = (tmp['data']*1)[np.sort(tmp['locs'+dhos[dho-1]]*1),:(rs[r-1])]
        self.r = rs[r-1]
        self.S = np.zeros((self.n))
        self.S[np.sort(tmp['locs'+dhos[dho-1]]*1)] = 1
        self.S = sparse.diags(self.S)
        self.S =  delete_rows_csr(self.S.tocsr(),np.where(self.S.diagonal() == 0))
        res = self.fit(data = self.data, r=self.r, S = self.S,verbose = verbose, fgrad = fgrad,par = par)
        np.savez(file = './fits/' + 'SA-' +mods[simmod-1]  + '-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz', par = res)
        return(True)
        
    def loadFit(self, simmod=None, dho=None, r=None, num=None, file = None):
        if file is None:
            mods = np.array(['SI','SA','NA1','NA2'])
            dhos = np.array(['100','10000','27000'])
            rs = np.array([1,10,100])
            file = './fits/' + mods[simmod-1] + '-SA-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz'
            print(file)
        fitmod = np.load(file)
        self.S = sparse.eye(self.n)
        par = fitmod['par']*1
        self.kappa = par[0]
        self.gamma = par[1]
        self.vx = par[2]
        self.vy = par[3]
        self.vz = par[4]
        self.rho1 = par[5]
        self.rho2 = par[6]
        self.tau = par[7]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        Hs = self.getH()
        Dk =  sparse.diags([np.exp(self.kappa)]*self.n) 
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = self.cholesky(self.Q)
        #self.mvar = rqinv(self.Q).diagonal()

    def sample(self,n = 1):
        z = np.random.normal(size = self.n*n).reshape(self.n,n)
        data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n*n).reshape(self.n,n)*np.exp(self.sigma)
        return(data)

    def sim(self):
        if not self.loaded:
            self.load()
        mods = []
        for file in os.listdir("./simulations/"):
            if file.startswith("SA-"):
                mods.append(int(file.split("-")[1].split(".")[0]))
        if not mods:
            num = 1
        else:
            num = max(mods) + 1 
        self.data = self.sample(n=100)
        np.savez('./simulations/SA-'+ str(num) +'.npz', data = self.data, locs100 = np.random.choice(np.arange(self.n), 100, replace = False), locs10000 = np.random.choice(np.arange(self.n), 10000, replace = False), locs27000 = np.arange(self.n))
        return(True)

    def setQ(self,par = None,S = None,simple = False):
        if par is None:
            assert(self.kappa is not None and self.gamma is not None and self.vx is not None and self.vy is not None and self.vz is not None and self.rho1 is not None and self.rho2 is not None and self.sigma is not None)
        else:
            self.kappa = par[0]
            self.gamma = par[1]
            self.vx = par[2]
            self.vy = par[3]
            self.vz = par[4]
            self.rho1 = par[5]
            self.rho2 = par[6]
            self.tau = par[7]
            self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        if S is not None:
            self.S = S
        Hs = self.getH()
        Dk =  sparse.diags([np.exp(self.kappa)]*self.n)
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = self.cholesky(self.Q)
        if not simple:
            self.Q_fac = self.cholesky(self.Q)
            
            
    def getH(self,par=None,d=None):
        if par is None:
            par = np.hstack([self.gamma,self.vx,self.vy,self.vz,self.rho1,self.rho2])
        if d is None:
            vv = np.array([par[1],par[2],par[3]])
            vb1 = np.array([-par[2],par[1],0])/np.sqrt(par[1]**2+par[2]**2)
            vb2 = np.array([-par[3]*par[1],-par[3]*par[2], par[1]**2 + par[2]**2])/np.sqrt(par[3]**2*par[1]**2 + par[3]**2*par[2]**2 + (par[1]**2+par[2]**2)**2)
            ww = vb1*par[4] + vb2*par[5]
            H = np.diag([np.exp(par[0]),np.exp(par[0]),np.exp(par[0])]) + vv[:,np.newaxis]*vv[np.newaxis,:] + ww[:,np.newaxis]*ww[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 0: # gamma1
            H = np.diag([np.exp(par[0]),np.exp(par[0]),np.exp(par[0])]) + np.zeros((self.n,6,3,3))
        elif d == 1: # vx
            vv = np.array([par[1],par[2],par[3]])
            dv = np.array([1,0,0])
            vb1 = np.array([-par[2],par[1],0])/np.sqrt(par[1]**2+par[2]**2)
            vb2 = np.array([-par[3]*par[1],-par[3]*par[2], par[1]**2 + par[2]**2])/np.sqrt(par[3]**2*par[1]**2 + par[3]**2*par[2]**2 + (par[1]**2+par[2]**2)**2)
            ww = vb1*par[4] + vb2*par[5]
            c1 = (par[1]**2 + par[2]**2)**(3.0/2.0)
            dvb1 = np.array([par[2]*par[1],par[2]**2,0])/c1
            c2 = (par[3]**2*par[1]**2 + par[3]**2*par[2]**2 + (par[1]**2 + par[2]**2)**2)**(3.0/2.0)
            dvb2 = np.array([par[3]*par[1]**4 - par[3]**3*par[2]**2-par[3]*par[2]**4,
                             par[3]**3*par[1]*par[2]+2*par[3]*par[2]*par[1]**3 + 2*par[3]*par[2]**3*par[1],
                             par[3]**2*par[1]**3 + par[3]**2*par[1]*par[2]**2])/c2
            dw = par[4]*dvb1 + par[5]*dvb2
            H = 2*dv[:,np.newaxis]*vv[np.newaxis,:] + 2*dw[:,np.newaxis]*ww[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 2: # vy
            vv = np.array([par[1],par[2],par[3]])
            dv = np.array([0,1,0])
            vb1 = np.array([-par[2],par[1],0])/np.sqrt(par[1]**2+par[2]**2)
            vb2 = np.array([-par[3]*par[1],-par[3]*par[2], par[1]**2 + par[2]**2])/np.sqrt(par[3]**2*par[1]**2 + par[3]**2*par[2]**2 + (par[1]**2+par[2]**2)**2)
            ww = vb1*par[4] + vb2*par[5]
            c1 = (par[1]**2 + par[2]**2)**(3.0/2.0)
            dvb1 = np.array([-par[1]**2,-par[2]*par[1],0])/c1
            c2 = (par[3]**2*par[1]**2 + par[3]**2*par[2]**2 +  (par[1]**2 + par[2]**2)**2)**(3.0/2.0)
            dvb2 = np.array([par[3]**3*par[1]*par[2] + 2*par[3]*par[1]**3*par[2] + 2*par[3]*par[1]*par[2]**3,
                             par[3]*par[2]**4 - par[3]**3*par[1]**2 - par[3]*par[1]**4,
                             par[3]**2*par[2]**3 + par[3]**2*par[1]**2*par[2]])/c2
            dw = par[4]*dvb1 + par[5]*dvb2
            H = 2*dv[:,np.newaxis]*vv[np.newaxis,:] + 2*dw[:,np.newaxis]*ww[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 3: # vz
            vv = np.array([par[1],par[2],par[3]])
            dv = np.array([0,0,1])
            vb1 = np.array([-par[2],par[1],0])/np.sqrt(par[1]**2+par[2]**2)
            vb2 = np.array([-par[3]*par[1],-par[3]*par[2], par[1]**2 + par[2]**2])/np.sqrt(par[3]**2*par[1]**2 + par[3]**2*par[2]**2 + (par[1]**2+par[2]**2)**2)
            ww = vb1*par[4] + vb2*par[5]
            c1 = (par[1]**2 + par[2]**2)**(3.0/2.0)
            dvb1 = np.array([0,0,0])/c1
            c2 = (par[3]**2*par[1]**2 + par[3]**2*par[2]**2 + (par[1]**2 + par[2]**2)**2)**(3.0/2.0)
            dvb2 = np.array([-par[1]**5 - 2*par[1]**3*par[2]**2 - par[1]*par[2]**4,
                             -par[1]**4*par[2] - 2*par[1]**2*par[2]**3 - par[2]**5,
                             -par[3]*par[1]**4 - 2*par[3]*par[1]**2*par[2]**2 - par[3]*par[2]**4])/c2
            dw = par[4]*dvb1 + par[5]*dvb2
            H = 2*dv[:,np.newaxis]*vv[np.newaxis,:] + 2*dw[:,np.newaxis]*ww[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 4: # rho1
            vb1 = np.array([-par[2],par[1],0])/np.sqrt(par[1]**2+par[2]**2)
            vb2 = np.array([-par[3]*par[1],-par[3]*par[2], par[1]**2 + par[2]**2])/np.sqrt(par[3]**2*par[1]**2 + par[3]**2*par[2]**2 + (par[1]**2+par[2]**2)**2)
            ww = vb1*par[4] + vb2*par[5]
            dw = vb1 
            H = 2*dw[:,np.newaxis]*ww[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 5: # rho2
            vb1 = np.array([-par[2],par[1],0])/np.sqrt(par[1]**2+par[2]**2)
            vb2 = np.array([-par[3]*par[1],-par[3]*par[2], par[1]**2 + par[2]**2])/np.sqrt(par[3]**2*par[1]**2 + par[3]**2*par[2]**2 + (par[1]**2+par[2]**2)**2)
            ww = vb1*par[4] + vb2*par[5]
            dw = vb2
            H = 2*dw[:,np.newaxis]*ww[np.newaxis,:] + np.zeros((self.n,6,3,3))
        return(H)

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
        data  = self.data
        Hs = self.getH(par[1:7])
        Dk =  np.exp(par[0])*sparse.eye(self.n) 
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Q = A_mat.transpose()@self.iDv@A_mat
        Q_c = Q + self.S.transpose()@self.S*np.exp(par[7])
        Q_fac = self.cholesky(Q)
        Q_c_fac= self.cholesky(Q_c)
        if (Q_fac == -1) or (Q_c_fac == -1):
            if self.grad:
                return((self.like,self.jac))
            else:
                return(self.like)
        mu_c = Q_c_fac.solve_A(self.S.transpose()@data*np.exp(par[7]))
        if self.r == 1:
            data = data.reshape(-1,1)
            mu_c = mu_c.reshape(-1,1)
        if self.grad:
            if np.abs(self.likediff) < 0.001:
                Qinv = rqinv(Q)
                Qcinv = rqinv(Q_c)
            else:
                Qinv,Qcinv = self.simpleMvar(Q_fac,Q,Q_c_fac)

            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[7]/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c.transpose()@Q@mu_c).diagonal().sum() - np.exp(par[7])/2*((data - self.S@mu_c).transpose()@(data-self.S@mu_c)).diagonal().sum()
            g_noise = self.S.shape[0]*self.r/2 - 1/2*(Qcinv@self.S.transpose()@self.S*np.exp(par[7])).diagonal().sum()*self.r - 1/2*((data - self.S@mu_c).transpose()@(data - self.S@mu_c)).diagonal().sum()*np.exp(par[7])

            A_par = Dk@self.Dv
            Q_par = A_par.transpose()@self.iDv@A_mat + A_mat.transpose()@self.iDv@A_par
            g_kappa = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

            Hs_par = self.getH(par[1:7],d=0)
            A_par = - AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
            Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
            g_gamma = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

            Hs_par = self.getH(par[1:7],d=1)
            A_par = - AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
            Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
            g_vx = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

            Hs_par = self.getH(par[1:7],d=2)
            A_par = - AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
            Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
            g_vy = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

            Hs_par = self.getH(par[1:7],d=3)
            A_par = - AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
            Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
            g_vz = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

            Hs_par = self.getH(par[1:7],d=4)
            A_par = - AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
            Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
            g_rho1 = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

            Hs_par = self.getH(par[1:7],d=5)
            A_par = - AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
            Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
            g_rho2 = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()
            
            like = -like/(self.r * self.S.shape[0])
            jac = -np.array([g_kappa,g_gamma,g_vx,g_vy,g_vz,g_rho1,g_rho2,g_noise])/(self.r * self.S.shape[0])
            self.likediff = like - self.like
            self.like = like
            self.opt_steps = self.opt_steps + 1
            self.jac = jac
            if self.end is not None:
                np.savez(self.end + '.npz',par)
            if self.verbose:
                if self.truth is not None:
                    print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%(par[0]-self.truth[0]), "\u03B3 = %2.2f"%(par[1]-self.truth[1]),"vx = %2.2f"%(np.abs(par[2])-np.abs(self.truth[2])),"vy = %2.2f"%(np.abs(par[3])-np.abs(self.truth[3])),
                    "vz = %2.2f"%(np.abs(par[4])-np.abs(self.truth[4])),"\u03C1_1 = %2.2f"%(np.abs(par[5])-np.abs(self.truth[5])),"\u03C1_2 = %2.2f"%(np.abs(par[6])-np.abs(self.truth[6])), "\u03C3 = %2.2f"%(par[7]-self.truth[7]))
                else:
                    print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3 = %2.2f"%np.exp(par[1]),"vx = %2.2f"%par[2],"vy = %2.2f"%par[3],
                    "vz = %2.2f"%(par[4]),"\u03C1_1 = %2.2f"%(par[5]),"\u03C1_2 = %2.2f"%(par[6]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[7])))
            return((like,jac))
        else: 
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[7]/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c.transpose()@Q@mu_c).diagonal().sum() - np.exp(par[7])/2*((data - self.S@mu_c).transpose()@(data-self.S@mu_c)).diagonal().sum()
            like = -like/(self.r * self.S.shape[0])
            self.like = like
            self.opt_steps = self.opt_steps + 1
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3_1 = %2.2f"%np.exp(par[1]),"\u03B3_2 = %2.2f"%np.exp(par[7]),"\u03B3_3 = %2.2f"%np.exp(par[3]),
                "\u03C1_1 = %2.2f"%(par[4]),"\u03C1_2 = %2.2f"%(par[5]),"\u03C1_3 = %2.2f"%(par[6]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[7])))
            return(like)

    