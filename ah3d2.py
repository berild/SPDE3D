import ctypes
import numpy as np
from scipy import sparse


lib = ctypes.cdll.LoadLibrary('./libAH3D2.so')

def AH(M, N, K, H, hx, hy,hz):
    fnew = lib.AH_new
    fnew.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=4,shape = (M*N*K,6,3,3)), ctypes.c_double,ctypes.c_double,ctypes.c_double]
    fnew.restype = ctypes.c_void_p
    obj = fnew(M, N, K, H, hx, hy, hz)

    frow = lib.AH_Row
    frow.argtypes = [ctypes.c_void_p]
    frow.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*K*19,))
    row = frow(obj)
    
    fcol = lib.AH_Col
    fcol.argtypes = [ctypes.c_void_p]
    fcol.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*K*19,))
    col = fcol(obj)

    fval = lib.AH_Val
    fval.argtypes = [ctypes.c_void_p]
    fval.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (M*N*K*19,))
    val = fval(obj)
    res = sparse.csc_matrix((val, (row, col)), shape=(M*N*K, M*N*K))

    fdel = lib.AH_delete
    fdel.argtypes = [ctypes.c_void_p]
    fdel.restype = None
    fdel(obj)
    return(res)

# class AH(object):
#     def __init__(self, M, N, K, H, hx, hy,hz):
#         self.M = M
#         self.N = N
#         self.K = K
#         fnew = lib.AH_new
#         fnew.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=4,shape = (M*N*K,6,3,3)), ctypes.c_double,ctypes.c_double,ctypes.c_double]
#         fnew.restype = ctypes.c_void_p
#         self.obj = fnew(M, N, K, H, hx, hy, hz)

#     def Row(self):
#         frow = lib.AH_Row
#         frow.argtypes = [ctypes.c_void_p]
#         frow.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (self.M*self.N*self.K*19,))
#         return(frow(self.obj))

#     def Col(self):
#         fcol = lib.AH_Col
#         fcol.argtypes = [ctypes.c_void_p]
#         fcol.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (self.M*self.N*self.K*19,))
#         return(fcol(self.obj))

#     def Val(self):
#         fval = lib.AH_Val
#         fval.argtypes = [ctypes.c_void_p]
#         fval.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (self.M*self.N*self.K*19,))
#         return(fval(self.obj))

#     def Get(self):
#         val = self.Val()
#         row = self.Row()
#         col = self.Col()
#         return(sparse.csc_matrix((val, (row, col)), shape=(self.M*self.N*self.K, self.M*self.N*self.K)))

#     def __enter__(self):
#         return self

#     def __exit__(self,exc_type,exc_val,exc_tb):
#         fun = lib.AH_delete
#         fun.argtypes = [ctypes.c_void_p]
#         fun.restype = None
#         fun(self.obj)

#     def __del__(self):
#         fun = lib.AH_delete
#         fun.argtypes = [ctypes.c_void_p]
#         fun.restype = None
#         fun(self.obj)