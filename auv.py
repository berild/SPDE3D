import numpy as np
from scipy import sparse
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
importr("Matrix")
import tempfile
#inla = importr("INLA")
robj.r.source("rqinv.R")
DEFAULT_NUM_SAMPLES = 250 

# class auv:
#     def __init__(self,spde,mu,sigma):
#         """Initialize model

#         Args:
#             model (int, optional): Doesn't do anything. Defaults to 2.
#             reduce (bool, optional): Reduced grid size used if set to True. Defaults to False.
#             method (int, optional): If model should contain fixed effects on the SINMOD mean. Defaults to 1.
#             prev (bool, optional): Loading previous model (used to clear memory)
#         """
#         self.spde = spde
#         Q = spde.mod.Q.copy()
#         self.n = spde.mod.n
#         Stot = sparse.eye(self.n)
#         self.mu = mu
#         self.mu2 = np.hstack([np.zeros(self.n),0,1]).reshape(-1,1) 
#         self.mu3 = mu

#         Q.resize((self.n+2,self.n+2))
#         Q = Q.tolil()
#         Q[self.n,self.n] = 0.1
#         Q[self.n+1,self.n+1] = 1
#         self.Q = Q.tocsc()
        
#         Stot.resize((self.n,self.n+2))
#         Stot = Stot.tolil()
#         Stot[:,self.n] = np.ones(self.n)
#         Stot[:,self.n+1] = mu
#         self.Stot = Stot.tocsc()
        
#         self.Q_fac = spde.mod.cholesky(self.Q)
#         self.sigma = sigma
        

#     def sample(self,n = 1):
#         """Samples the GMRF. Only used to test.

#         Args:
#             n (int, optional): Number of realizations. Defaults to 1.
#         """
#         z = np.random.normal(size = (self.n+2)*n).reshape((self.n+2),n)
#         data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
#         data = data[:self.n,:] + self.mu3.reshape(-1,1) + np.random.normal(size = self.n*n).reshape(self.n,n)*self.sigma
#         return(data)

#     def update(self, data, keep=False):
#         """Update mean and precision of the GMRF given some measurements in the field.

#         Args:
#             rel ([k,1]-array): k number of measurements of the GMRF. (k>0).
#             ks ([k,]-array): k number of indicies describing the index of the measurment in the field. 
#         """
#         ks = data['idx'].to_numpy()
#         rel = data['data'].to_numpy().reshape(-1,1)
#         if ks.size>0:
#             S = self.Stot[ks,:]
#             self.Q = self.Q + S.transpose()@S*1/self.sigma**2
#             self.Q_fac.cholesky_inplace(self.Q)
#             self.mu2 = self.mu2 - self.Q_fac.solve_A(S.transpose().tocsc()@(S@self.mu2 - rel))*1/self.sigma**2
#             self.mu = (self.Stot@self.mu2)[:,0]
#         if keep:
#             self.Q_k = self.Q.copy()
#             self.mu2_k = self.mu2
#             self.mu_k = self.mu

#     def loadKeep(self):
#         self.Q = self.Q_k.copy()
#         self.mu2 = self.mu2_k
#         self.mu = self.mu_k
#         self.Q_fac.cholesky_inplace(self.Q_k)

#     def mvar(self, simple = False, n = DEFAULT_NUM_SAMPLES):
#         if not simple:
#             tshape = self.Q.shape
#             Q = self.Q.copy().tocoo()
#             r = Q.row
#             c = Q.col
#             v = Q.data
#             tmpQinv =  np.array(robj.r.rqinv(robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))
#             return(sparse.csc_matrix((np.array(tmpQinv[:,2],dtype = "float32"), (np.array(tmpQinv[:,0],dtype="int32"), np.array(tmpQinv[:,1],dtype="int32"))), shape=tshape).diagonal()+self.sigma**2)
#         else:
#             return(self.sample(n=DEFAULT_NUM_SAMPLES).var(axis = 1))

class auv:
    def __init__(self,spde,mu,sigma):
        """Initialize model

        Args:
            model (int, optional): Doesn't do anything. Defaults to 2.
            reduce (bool, optional): Reduced grid size used if set to True. Defaults to False.
            method (int, optional): If model should contain fixed effects on the SINMOD mean. Defaults to 1.
            prev (bool, optional): Loading previous model (used to clear memory)
        """
        self.spde = spde
        self.Q = spde.mod.Q.copy()
        self.n = spde.mod.n
        self.Stot = sparse.eye(self.n).tocsc()
        self.mu = mu
        
        self.Q_fac = spde.mod.cholesky(self.Q)
        self.sigma = sigma
        self.Q_k = self.Q.copy()
        self.mu_k = self.mu
        

    def sample(self,n = 1):
        """Samples the GMRF. Only used to test.

        Args:
            n (int, optional): Number of realizations. Defaults to 1.
        """
        z = np.random.normal(size = self.n*n).reshape(self.n,n)
        data = self.mu[:,np.newaxis] + self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) # + np.random.normal(size = self.n*n).reshape(self.n,n)*self.sigma
        return(data)

    def update(self, data, keep=False):
        """Update mean and precision of the GMRF given some measurements in the field.

        Args:
            rel ([k,1]-array): k number of measurements of the GMRF. (k>0).
            ks ([k,]-array): k number of indicies describing the index of the measurment in the field. 
        """
        ks = data['idx'].to_numpy()
        rel = data['data'].to_numpy().reshape(-1,1)
        self.mu = self.mu.reshape(-1,1)
        if ks.size>0:
            S = self.Stot[ks,:]
            self.Q = self.Q + S.transpose()@S*1/self.sigma**2
            self.Q_fac.cholesky_inplace(self.Q)
            self.mu = self.mu - self.Q_fac.solve_A(S.transpose().tocsc())@(S@self.mu - rel)*1/self.sigma**2
        self.mu = self.mu.reshape(-1)
        if keep:
            self.Q_k = self.Q.copy()
            self.mu_k = self.mu

    def loadKeep(self):
        self.Q = self.Q_k.copy()
        self.mu = self.mu_k
        self.Q_fac.cholesky_inplace(self.Q_k)

    def var(self, simple = False, n = DEFAULT_NUM_SAMPLES):
        if not simple:
            tshape = self.Q.shape
            Q = self.Q.copy().tocoo()
            r = Q.row
            c = Q.col
            v = Q.data
            tmpQinv =  np.array(robj.r.rqinv(robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))
            return(sparse.csc_matrix((np.array(tmpQinv[:,2],dtype = "float32"), (np.array(tmpQinv[:,0],dtype="int32"), np.array(tmpQinv[:,1],dtype="int32"))), shape=tshape).diagonal())#+self.sigma**2)
        else:
            return(self.sample(n=DEFAULT_NUM_SAMPLES).var(axis = 1))