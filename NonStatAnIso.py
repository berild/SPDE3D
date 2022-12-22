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


class NonStatAnIso:
    #mod4: kappa(0:27), gamma(27:54), vx(54:81), vy(81:108), vz(108:135), rho1(135:162), rho2(162:189), sigma(189)
    def __init__(self,grid=None,par=None):
        assert par is None or (par.size==190 if par is not None else False)
        self.grid = Grid() if grid is None else grid
        self.n = self.grid.M*self.grid.N*self.grid.P
        self.kappa = np.array([0.5]*27) if par is None else par[0:27]
        self.gamma = np.array([0.5]*27) if par is None else par[27:54]
        self.vx = np.array([0.5]*27) if par is None else par[54:81]
        self.vy = np.array([0.5]*27) if par is None else par[81:108]
        self.vz = np.array([0.5]*27) if par is None else par[108:135]
        self.rho1 = np.array([0.5]*27) if par is None else par[135:162]
        self.rho2 = np.array([0.5]*27) if par is None else par[162:189]
        self.tau = 0.5 if par is None else par[189]
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
        self.jac = np.array([-100]*190)
        self.loaded = False
        self.truth = None
        self.end = None

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
        simmod = np.load("./simmodels/NA.npz")
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
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa)))
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        self.Q = A_mat.transpose()@self.iDv@A_mat
        if not simple:
            self.Q_fac = self.cholesky(self.Q)
            assert(self.Q_fac != -1)

    def fit(self,data, r, S,verbose = False, fgrad = True, par = None,end = None):
        #mod4: kappa(0:27), gamma(27:54), vx(54:81), vy(81:108), vz(108:135), rho1(135:162), rho2(162:189), sigma(189)
        if par is None:
            par = np.array([-0.5]*190)
        self.data = data
        self.r = r
        self.S = S
        self.opt_steps = 0
        self.grad = fgrad
        self.verbose = verbose
        self.likediff = 1000
        self.like = -10000
        self.end = end
        if self.grad:
            print("Running...")
            res = minimize(self.logLike, x0 = par,jac = True, method = "BFGS",tol = 1e-4)
            res = res['x']
        else:    
            res = minimize(self.logLike, x0 = par, tol = 1e-4)
            res = res['x']
        self.kappa = res[0:27]
        self.gamma = res[27:54]
        self.vx = res[54:81]
        self.vy = res[81:108]
        self.vz = res[108:135]
        self.rho1 = res[135:162]
        self.rho2 = res[162:189]
        self.tau = res[189]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        return(res)

    def fitTo(self,simmod,dho,r,num,verbose = False, fgrad = True, par = None):
        if simmod == 4:
            tmp = np.load("./simmodels/NA.npz")
            self.truth = np.hstack([tmp['kappa'],tmp['gamma'],tmp['vx'],tmp['vy'],tmp['vz'],tmp['rho1'],tmp['rho2'],np.log(1/np.exp(tmp['sigma'])**2)])
        if par is None:
            par = np.load('./simmodels/initNA.npy')
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
        np.savez('./fits/' + 'NA-' + mods[simmod-1] + '-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz', par = res)
        return(True)

    def loadFit(self, simmod=None, dho=None, r=None, num=None, file = None):
        if file is None:
            mods = np.array(['SI','SA','NI','NA'])
            dhos = np.array(['100','10000','27000'])
            rs = np.array([1,10,100])
            file = './fits/NA-' + mods[simmod-1] + '-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz'
            print(file)
        fitmod = np.load(file)
        self.S = sparse.eye(self.n)
        par =fitmod['par']*1
        self.kappa = par[0:27]
        self.gamma = par[27:54]
        self.vx = par[54:81]
        self.vy = par[81:108]
        self.vz = par[108:135]
        self.rho1 = par[135:162]
        self.rho2 = par[162:189]
        self.tau = par[189]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        Hs = self.getH()
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa))) 
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = self.cholesky(self.Q)
        #self.mvar = rqinv(self.Q).diagonal()

    # def sample(self,n = 1, Q_fac = None, sigma = None, simple = False):
    #     if Q_fac is None:
    #         Q_fac = self.Q_fac
    #     z = np.random.normal(size = self.n*n).reshape(self.n,n)
    #     if simple:
    #         data = Q_fac.apply_Pt(Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
    #     else:
    #         if sigma is None:
    #             sigma = self.sigma 
    #         data = Q_fac.apply_Pt(Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n*n).reshape(self.n,n)*np.exp(sigma)
    #     if n ==1:
    #         data = data.reshape(self.n)
    #     return(data)
    def sample(self,n = 1,simple = False):
        z = np.random.normal(size = self.n*n).reshape(self.n,n)
        if simple:
            data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
        else:
            data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n*n).reshape(self.n,n)*np.exp(self.sigma)
        return(data)

    def sim(self):
        if not self.loaded:
            self.load()
        mods = []
        for file in os.listdir("./simulations/"):
            if file.startswith("NA-"):
                mods.append(int(file.split("-")[1].split(".")[0]))
        if not mods:
            num = 1
        else:
            num = max(mods) + 1 
        self.data = self.sample(n=100)
        np.savez('./simulations/NA-'+ str(num) +'.npz', data = self.data, locs100 = np.random.choice(np.arange(self.n), 100, replace = False), locs10000 = np.random.choice(np.arange(self.n), 10000, replace = False), locs27000 = np.arange(self.n))
        return(True)

    def setQ(self,par = None,S = None, simple = False):
        if par is None:
            assert(self.kappa is not None and self.gamma is not None and self.vx is not None and self.vy is not None and self.vz is not None and self.rho1 is not None and self.rho2 is not None and self.sigma is not None)
        else:
            self.kappa = par[0:27]
            self.gamma = par[27:54]
            self.vx = par[54:81]
            self.vy = par[81:108]
            self.vz = par[108:135]
            self.rho1 = par[135:162]
            self.rho2 = par[162:189]
            self.tau = par[189]
            self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        if S is not None:
            self.S = S
        Hs = self.getH()
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa))) 
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        self.Q = A_mat.transpose()@self.iDv@A_mat
        if not simple:
            self.Q_fac = self.cholesky(self.Q)

    def getH(self,gamma = None,vx = None,vy = None, vz = None,rho1 = None,rho2 = None,d=None,grad = False):
        if gamma is None and vx is None and vy is None and vz is None and rho1 is None and rho2 is None:
            gamma = self.gamma
            vx = self.vx
            vy = self.vy
            vz = self.vz
            rho1 = self.rho1
            rho2 = self.rho2
        if not grad:
            pg = np.exp(self.grid.evalBH(par = gamma))
            vv = np.stack([self.grid.evalBH(par = vx),self.grid.evalBH(par = vy),self.grid.evalBH(par = vz)],axis=2)
            vb1 = np.stack([-vv[:,:,1],vv[:,:,0],vv[:,:,2]*0],axis=2)/np.sqrt(vv[:,:,0]**2 + vv[:,:,1]**2)[:,:,np.newaxis]
            vb2 = np.stack([-vv[:,:,2]*vv[:,:,0],-vv[:,:,2]*vv[:,:,1],vv[:,:,0]**2 + vv[:,:,1]**2],axis=2)/np.sqrt(vv[:,:,2]**2*vv[:,:,0]**2 + vv[:,:,2]**2*vv[:,:,1]**2+(vv[:,:,0]**2 + vv[:,:,1]**2)**2)[:,:,np.newaxis]
            ww = vb1*self.grid.evalBH(par = rho1)[:,:,np.newaxis] + vb2*self.grid.evalBH(par = rho2)[:,:,np.newaxis]
            H = (np.eye(3)*(np.stack([pg,pg,pg],axis=2))[:,:,:,np.newaxis]) + vv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:] + ww[:,:,:,np.newaxis]*ww[:,:,np.newaxis,:]
            return(H)
        else:
            dpar = np.zeros(27)
            dpar[d] = 1
            pg = np.exp(self.grid.evalBH(par = gamma))
            H_gamma = np.eye(3)*(np.stack([self.grid.bsH[:,:,d]*pg,self.grid.bsH[:,:,d]*pg,self.grid.bsH[:,:,d]*pg],axis=2)[:,:,:,np.newaxis])

            vv = np.stack([self.grid.evalBH(par = vx),self.grid.evalBH(par = vy),self.grid.evalBH(par = vz)],axis=2)
            vb1 = np.stack([-vv[:,:,1],vv[:,:,0],vv[:,:,2]*0],axis=2)/np.sqrt(vv[:,:,0]**2 + vv[:,:,1]**2)[:,:,np.newaxis]
            vb2 = np.stack([-vv[:,:,2]*vv[:,:,0],-vv[:,:,2]*vv[:,:,1],vv[:,:,0]**2 + vv[:,:,1]**2],axis=2)/np.sqrt(vv[:,:,2]**2*vv[:,:,0]**2 + vv[:,:,2]**2*vv[:,:,1]**2+(vv[:,:,0]**2 + vv[:,:,1]**2)**2)[:,:,np.newaxis]
            ww = vb1*self.grid.evalBH(par = rho1)[:,:,np.newaxis] + vb2*self.grid.evalBH(par = rho2)[:,:,np.newaxis]
            
            dv = np.stack([self.grid.evalBH(par = dpar),self.grid.evalBH(par = np.zeros(27)),self.grid.evalBH(par = np.zeros(27))],axis = 2)
            c1 = (vv[:,:,0]**2 + vv[:,:,1]**2)**(3.0/2.0)
            c2 = (vv[:,:,2]**2*vv[:,:,0]**2 + vv[:,:,2]**2*vv[:,:,1]**2+(vv[:,:,0]**2 + vv[:,:,1]**2)**2)**(3.0/2.0)
            dvb1 = np.stack([vv[:,:,1]*vv[:,:,0],
                             vv[:,:,1]**2,
                             vv[:,:,2]*0],axis=2)*(dv[:,:,0]/c1)[:,:,np.newaxis]
            dvb2 = np.stack([vv[:,:,2]*vv[:,:,0]**4 - vv[:,:,2]**3*vv[:,:,1]**2 - vv[:,:,2]*vv[:,:,1]**4,
                             vv[:,:,2]**3*vv[:,:,0]*vv[:,:,1] + 2*vv[:,:,2]*vv[:,:,1]*vv[:,:,0]**3 + 2*vv[:,:,2]*vv[:,:,1]**3*vv[:,:,0],
                             vv[:,:,2]**2*vv[:,:,0]**3 + vv[:,:,2]**2*vv[:,:,0]*vv[:,:,1]**2],axis=2)*(dv[:,:,0]/c2)[:,:,np.newaxis]
            dw = dvb1*self.grid.evalBH(par = rho1)[:,:,np.newaxis] + dvb2*self.grid.evalBH(par = rho2)[:,:,np.newaxis]
            H_vx = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:] + dw[:,:,:,np.newaxis]*ww[:,:,np.newaxis,:] + ww[:,:,:,np.newaxis]*dw[:,:,np.newaxis,:]

            dv = np.stack([self.grid.evalBH(par = np.zeros(27)),self.grid.evalBH(par = dpar),self.grid.evalBH(par = np.zeros(27))],axis = 2)
            dvb1 = np.stack([-vv[:,:,0]**2,
                             -vv[:,:,0]*vv[:,:,1],
                             vv[:,:,2]*0],axis=2)*(dv[:,:,1]/c1)[:,:,np.newaxis]
            dvb2 = np.stack([vv[:,:,2]**3*vv[:,:,0]*vv[:,:,1] + 2*vv[:,:,2]*vv[:,:,0]**3*vv[:,:,1] + 2*vv[:,:,2]*vv[:,:,0]*vv[:,:,1]**3,
                             vv[:,:,2]*vv[:,:,1]**4 - vv[:,:,2]**3*vv[:,:,0]**2 - vv[:,:,2]*vv[:,:,0]**4,
                             vv[:,:,2]**2*vv[:,:,1]**3 + vv[:,:,2]**2*vv[:,:,0]**2*vv[:,:,1]],axis=2)*(dv[:,:,1]/c2)[:,:,np.newaxis]
            dw = dvb1*self.grid.evalBH(par = rho1)[:,:,np.newaxis] + dvb2*self.grid.evalBH(par = rho2)[:,:,np.newaxis]
            H_vy = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:] + dw[:,:,:,np.newaxis]*ww[:,:,np.newaxis,:] + ww[:,:,:,np.newaxis]*dw[:,:,np.newaxis,:]
            
            dv = np.stack([self.grid.evalBH(par = np.zeros(27)),self.grid.evalBH(par = np.zeros(27)),self.grid.evalBH(par = dpar)],axis = 2)
            dvb2 = np.stack([-vv[:,:,0]**5 - 2*vv[:,:,0]**3*vv[:,:,1]**2 - vv[:,:,0]*vv[:,:,1]**4,
                             -vv[:,:,0]**4*vv[:,:,1] - 2*vv[:,:,0]**2*vv[:,:,1]**3 - vv[:,:,1]**5,
                             -vv[:,:,2]*vv[:,:,0]**4 - 2*vv[:,:,2]*vv[:,:,0]**2*vv[:,:,1]**2 - vv[:,:,2]*vv[:,:,1]**4],axis=2)*(dv[:,:,2]/c2)[:,:,np.newaxis]
            dw = dvb2*self.grid.evalBH(par = rho2)[:,:,np.newaxis]
            H_vz = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:] + dw[:,:,:,np.newaxis]*ww[:,:,np.newaxis,:] + ww[:,:,:,np.newaxis]*dw[:,:,np.newaxis,:]

            dw = vb1*self.grid.evalBH(par=dpar)[:,:,np.newaxis] 
            H_rho1 = ww[:,:,:,np.newaxis]*dw[:,:,np.newaxis,:] + dw[:,:,:,np.newaxis]*ww[:,:,np.newaxis,:]

            
            dw = vb2*self.grid.evalBH(par = dpar)[:,:,np.newaxis] 
            H_rho2 = ww[:,:,:,np.newaxis]*dw[:,:,np.newaxis,:] + dw[:,:,:,np.newaxis]*ww[:,:,np.newaxis,:]
            return((H_gamma,H_vx,H_vy,H_vz,H_rho1,H_rho2))

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
        #mod4: kappa(0:27), gamma(27:54), vx(54:81), vy(81:108), vz(108:135), rho1(135:162), rho2(162:189), sigma(189)
        data  = self.data
        Hs = self.getH(gamma = par[27:54],vx = par[54:81], vy = par[81:108], vz = par[108:135],rho1=par[135:162],rho2=par[162:189]) 
        lkappa = self.grid.evalB(par = par[0:27])
        Dk =  sparse.diags(np.exp(lkappa)) 
        A_mat = self.Dv@Dk - AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Q = A_mat.transpose()@self.iDv@A_mat
        Q_c = Q + self.S.transpose()@self.S*np.exp(par[189])
        Q_fac = self.cholesky(Q)
        Q_c_fac= self.cholesky(Q_c)
        if (Q_fac == -1) or (Q_c_fac == -1):
            if self.grad:
                return((self.like,self.jac))
            else:
                return(self.like)
        mu_c = Q_c_fac.solve_A(self.S.transpose()@data*np.exp(par[189]))
        if self.r == 1:
            data = data.reshape(-1,1)
            mu_c = mu_c.reshape(-1,1)
        if self.grad:
            if np.abs(self.likediff) < 0.001:
                Qinv =  rqinv(Q) 
                Qcinv = rqinv(Q_c)
            else:
                Qinv,Qcinv = self.simpleMvar(Q_fac,Q,Q_c_fac)

            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[189]/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c.transpose()@Q@mu_c).diagonal().sum() - np.exp(par[189])/2*((data - self.S@mu_c).transpose()@(data-self.S@mu_c)).diagonal().sum()
            g_par = np.zeros(190)
            g_par[189] = self.S.shape[0]*self.r/2 - 1/2*(Qcinv@self.S.transpose()@self.S*np.exp(par[189])).diagonal().sum()*self.r - 1/2*((data - self.S@mu_c).transpose()@(data - self.S@mu_c)).diagonal().sum()*np.exp(par[189])

            for i in range(27):
                A_par = self.Dv@sparse.diags(self.grid.bs[:,i]*np.exp(lkappa))
                Q_par = A_par.transpose()@self.iDv@A_mat + A_mat.transpose()@self.iDv@A_par
                g_par[i] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

                H_gamma, H_vx, H_vy, H_vz,H_rho1, H_rho2 = self.getH(gamma = par[27:54],vx = par[54:81], vy = par[81:108], vz = par[108:135],rho1=par[135:162],rho2=par[162:189],d=i,grad=True) 

                A_par = - AH(self.grid.M,self.grid.N,self.grid.P,H_gamma,self.grid.hx,self.grid.hy,self.grid.hz)
                Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
                g_par[27 + i] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()

                A_par = - AH(self.grid.M,self.grid.N,self.grid.P,H_vx,self.grid.hx,self.grid.hy,self.grid.hz)
                Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
                g_par[54 + i] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()
                
                A_par = - AH(self.grid.M,self.grid.N,self.grid.P,H_vy,self.grid.hx,self.grid.hy,self.grid.hz)
                Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
                g_par[81 + i] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()
                  
                A_par = - AH(self.grid.M,self.grid.N,self.grid.P,H_vz,self.grid.hx,self.grid.hy,self.grid.hz)
                Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
                g_par[108 + i] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()
                
                A_par = - AH(self.grid.M,self.grid.N,self.grid.P,H_rho1,self.grid.hx,self.grid.hy,self.grid.hz)
                Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
                g_par[135 + i] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()
                
                A_par = - AH(self.grid.M,self.grid.N,self.grid.P,H_rho2,self.grid.hx,self.grid.hy,self.grid.hz)
                Q_par = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
                g_par[162 + i] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r  - 1/2*((mu_c.transpose()@Q_par)@mu_c).diagonal().sum()
                
            like =  -like/(self.S.shape[0]*self.r)
            self.likediff = like - self.like
            self.like = like
            jac =  -g_par/(self.S.shape[0]*self.r)
            self.opt_steps = self.opt_steps + 1
            if self.end is not None:
                np.savez(self.end + '.npz',par)
            if self.verbose:
                if self.truth is not None:
                    print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%(np.mean(par[0:27]-self.truth[:27])), "\u03B3 = %2.2f"%(np.mean(par[27:54]-self.truth[27:54])),"vx = %2.2f"%(np.mean(np.abs(par[54:81])-np.abs(self.truth[54:81]))),"vy = %2.2f"%(np.mean(np.abs(par[81:108])-np.abs(self.truth[81:108]))),
                    "vz = %2.2f"%(np.mean(np.abs(par[108:135])-np.abs(self.truth[108:135]))),"\u03C1_1 = %2.2f"%(np.mean(np.abs(par[135:162])-np.abs(self.truth[135:162]))),"\u03C1_2 = %2.2f"%(np.mean(np.abs(par[162:189])-np.abs(self.truth[162:189]))), "\u03C3 = %2.2f"%(par[189]-self.truth[189]))
                else:  
                    print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like))
            return((like,jac))
        else: 
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[189]/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c.transpose()@Q@mu_c).diagonal().sum() - np.exp(par[189])/2*((data - self.S@mu_c).transpose()@(data-self.S@mu_c)).diagonal().sum()
            like = -like/(self.S.shape[0]*self.r)
            self.like = like
            self.opt_steps = self.opt_steps + 1
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like))
            return(like)
