import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

H = 7000.   # Scale height for log-pressure coordinates
cp = 1004.   # J / K kg specific heat at constant pressure
R = 287.    # J / K kg specific gas constant for dry air


class BaseState():
    '''  A simple PDE solver for coupled set of advection/reaction equations
    for the evolution of QBO temperature, ozone, N2O and NOx anomalies given
    a profile of background upwelling and anomalous QBO-related upwelling.
    '''
    def __init__(self, zb, zt, Nz = 800):
        # Grid parameters
        self.__dict__['zb'] = zb
        self.__dict__['zt'] = zt
        self.__dict__['Lz'] = zt - zb
        self.__dict__['Nz'] = Nz

        self.initialize()

        # Upwelling (background, perterbation)
        self.__dict__['w0'] = 0.0003 + 0.0 * self.zs
        self.__dict__['wp'] = 0.0001 + 0j + 0.0 * self.zs

        self.__dict__['S0']     = 0.012  + 0.0 * self.zs
        self.__dict__['dO3dz']  = 0.0005 + 0.0 * self.zs
        self.__dict__['dN2Odz'] = 0.0005 + 0.0 * self.zs
        self.__dict__['dNOxdz'] = 0.0005 + 0.0 * self.zs

        self.__dict__['omega']   = 2 * np.pi / 840. / 86400.

        # Temperature coeffs
        self.__dict__['aT']   = 0.1 / 86400  + 0.0 * self.zs
        self.__dict__['aO3']  = 0.3 / 86400  + 0.0 * self.zs

        # Ozone coeffs
        self.__dict__['dT']   = 0.01 / 86400 + 0.0 * self.zs
        self.__dict__['dO3']  = 0.01 / 86400 + 0.0 * self.zs
        self.__dict__['dNOx'] = 0.01 / 86400 + 0.0 * self.zs

        # N2O coeffs
        self.__dict__['gN2O']  = 0.01 / 86400 + 0.0 * self.zs
        self.__dict__['gNOx']  = 0.0  / 86400 + 0.0 * self.zs

        # NOx coeffs
        self.__dict__['eN2O']  = 0.01 / 86400 + 0.0 * self.zs
        self.__dict__['eNOx']  = 0.0  / 86400 + 0.0 * self.zs

        self.__dict__['T0'  ] = 0.
        self.__dict__['O30' ] = 0.
        self.__dict__['N2O0'] = 0.
        self.__dict__['NOx0'] = 0.

    def __setattr__(self, name, value):
        if name in self.__dict__.keys(): self.__dict__[name] = value
        else: raise ValueError('%s has no attribute "%s".' % (self, name))
        self.initialize()

    def initialize(self):
        self.__dict__['zs'] = np.linspace(self.zb, self.zt, self.Nz + 1)
        self.__dict__['dz'] = self.Lz / (self.Nz + 1)

        ps = 1000 * np.exp(-self.zs / H)
        self.__dict__['ps'] = ps

    def solve(self):
        def Dz(c): return sparse.diags([-c[1:], c], [-1, 0], shape=(self.Nz, self.Nz), dtype = np.complex64, format = 'csr')
        def C(c):  return sparse.diags([c], [0], shape=(self.Nz, self.Nz), dtype = np.complex64, format = 'csr')

        dz = self.dz

        # Lower boundary conditions
        T0   = self.T0
        O30  = self.O30
        N2O0 = self.N2O0
        NOx0 = self.NOx0

        # Temperature coefficients
        tz = self.w0
        tc = 1j * self.omega + self.aT + self.w0 * R / (cp * H)
        to = -self.aO3

        tf = -self.S0 * self.wp

        # Ozone coefficients
        oz  = self.w0
        oc  = 1j * self.omega + self.dO3
        ot  = -self.dT
        ox  = -self.dNOx

        of = -self.dO3dz * self.wp

        # N2O coefficients
        nz  = self.w0
        nc  = 1j * self.omega + self.gN2O
        nx  = -self.gNOx

        nf = -self.dN2Odz * self.wp

        # NOx coefficients
        xz  = self.w0
        xc  = 1j * self.omega + self.eNOx
        xn  = -self.eN2O

        xf = -self.dNOxdz * self.wp

        # Operator blocks
        Ltt = Dz(tz[1:]) + C(tc[1:] * dz)
        Lto = C(to[1:] * dz)

        Lot = C(ot[1:] * dz)
        Loo = Dz(oz[1:]) + C(oc[1:] * dz)
        Lox = C(ox[1:] * dz)

        Lnn = Dz(nz[1:]) + C(nc[1:] * dz)
        Lnx = C(nx[1:] * dz)

        Lxx = Dz(xz[1:]) + C(xc[1:] * dz)
        Lxn = C(xn[1:] * dz)

        # Null matrix
        Z = None

        L = sparse.bmat([[Ltt, Lto, Z  , Z], 
                                [Lot, Loo, Z  , Lox], 
                                [Z  , Z  , Lnn, Lnx], 
                                [Z  , Z  , Lxn, Lxx]], 
                               format='csr', dtype = np.complex64)

        # Forcing and lower boundary conditions
        Ft    = tf[1:] * dz
        Ft[0] += tz[0] * T0

        FO3    = of[1:] * dz
        FO3[0] += oz[0] * O30

        FN2O    = nf[1:] * dz
        FN2O[0] += nz[0] * N2O0

        FNOx    = xf[1:] * dz
        FNOx[0] += xz[0] * NOx0

        F = np.concatenate([Ft, FO3, FN2O, FNOx])

        # Solve system
        sln = spsolve(L, F)

        # Extract components
        T   = np.concatenate([ [T0],  sln[:self.Nz]])
        O3  = np.concatenate([[O30],  sln[self.Nz:2*self.Nz]])
        NO2 = np.concatenate([[N2O0], sln[2*self.Nz:3*self.Nz]])
        NOx = np.concatenate([[NOx0], sln[3*self.Nz:]])

        return T, O3, NO2, NOx
