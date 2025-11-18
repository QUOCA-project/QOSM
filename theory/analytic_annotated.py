# ---
# jupyter:
#   jupytext:
#     cell_markers: '{{{,}}}'
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Plotting routines for analytic solutions given in shared overleaf document

import numpy as np
from matplotlib import pyplot as plt

from scipy import special as sp
from scipy import sparse

# Numpy functions work with complex input, scipy special polynomials weren't (at least how I was doing it)
from numpy.polynomial import hermite_e

import pygeode as pyg

# Fixed Constants
Omega = 7.29e-5 
a = 6317e3
N2 = 4e-4
beta = 2*Omega / a
H = 7e3
R = 287.
om = 2*np.pi/(28*30*86400.)   # QBO frequency in Hz (period = 840 days)

# Figure path (plots are saved here)
figpath = '../figs/'

# Analytic n=0 Hermite–Gaussian solution: computes the basic forcing,
# zonal wind, and temperature shapes using ℓ² and A = (iω+α)/(iω). Implements
# the closed-form expressions corresponding to eqs. (7.26)–(7.28).
def plot_fields(tau = 10, D = 2e3):
   '''Plots solution to forcing by a gaussian that projects onto a single eigenfunction.'''
   
   if tau == 0.: al = 0.
   else: al = 1 / (tau * 86400.)

   A = (1j * om + al) / (1j * om)

   l2 = np.sqrt(N2/A) * D / (2 * beta)

   print('A:', np.absolute(A), A / np.absolute(A))
   print('L; l2 normed:', np.sqrt(2*np.absolute(l2))*1e-3, l2 / np.absolute(l2))

   lat = pyg.regularlat(181)
   y = (a*lat*np.pi/180.).rename('y')

   F = pyg.exp(-y**2 / (4*l2))
   u = F * (1 - y**2 / (6 * l2))
   T = F * (1 - y**2 / (2 * l2)) / 3.

   F = F.rename('F')
   u = u.rename('u')
   T = T.rename('T')

   pyg.showlines([F.real(), u.real(), T.real()], fig=1)
   #pyg.showlines([F.imag(), u.imag(), T.imag()], fig=1)

# Decomposes a Gaussian forcing into Hermite modes using the expansion in Sec. 7:
# F(x) = B Σ (Cⁿ/n!) He₂ₙ(x) e^{-x²/4}. Computes B and C from the forcing width
# and plots each Hermite component and their reconstruction against the original Gaussian.
def plot_gaussian_exp(r2, N=4):
   '''Plots decomposition of arbitraty gaussian into Hermite functions'''
   x = np.linspace(-5, 5, 1001) + 0.0j

   L2 = 1/r2

   G = np.exp(-x**2 / (2 * L2))

   g = np.exp(-x**2 / 4.)

   B = (0.5 + 1./L2)**-0.5
   C = (L2 - 2) / (4 + 2 * L2)

   print(f'r2 {r2:.2f}; B {B:.2f}, C {C:.2f}')
   # term n
   gn = lambda n: C**n / sp.factorial(n) * B

   f = 0j * x

   plt.ioff()

   fig = plt.figure(2)
   fig.clf()

   ax = fig.add_subplot(111)

   ax.plot(x, G, 'k--', lw = 2., label = 'f')

   for n in range(N):
      coef = [gn(n) if i == 2*n else 0. for i in range(2*N + 1)]
      gp = hermite_e.hermeval(x, coef) * g
      ax.plot(x.real, gp.real, lw=1., label = f'n = {n}')
      f += gp

   ax.plot(x, f, 'k', lw = 2., label = 'sum')
   ax.set(ylim = (-0.5, 1.5))

   plt.ion()

   plt.show()
   plt.draw()


# Computes the temperature response by projecting a Gaussian forcing onto Hermite
# modes, constructing T(x) = Σ Tₙ He₂ₙ(x)e^{-x²/4}, and comparing the
# analytic Hermite sum with the asymptotic and numerical solutions.
def plot_T_sum(L, N=4, tau = 10, D = 2e3, F = 0.01 / 86400., savefig = False):
   '''Plots temperature response to an arbitrary gaussian forcing, decomposed by eigenfunction.'''
   
   # Numerical solution for comparison
   ds = solve_T_num(L, tau, D, F)

   Tapp = solve_T_approx(L, tau, D, F)

   if tau == 0: 
      taulbl = r'$\tau$: None'
      al = 0.
   else: 
      taulbl = r'$\tau$'f': {tau} d'
      al = 1 / (tau * 86400.)

   A = (1j * om + al) / (1j * om)

   l2 = np.sqrt(N2/A) * D / (2 * beta)

   r2 = l2 / L**2
   l = np.sqrt(l2)

   #Tfac = 1j / (1j*om) * H / R * np.sqrt(A * N2) * F / 4.
   Tc = -1j / (1j*om + al) * H / R * np.sqrt(A * N2) * F / 4.

   print(f'A {A:.2f}; r2 {r2:.2f}; l {l:.2f}, Tc {Tc}')

   lat = pyg.regularlat(181)[:]
   y = (a*lat*np.pi/180.)

   x = y / l

   #x = np.linspace(-8, 8, 1001)

   G = np.exp(-r2 * x**2 / 2)

   g = np.exp(-x**2 / 4.)

   B = (0.5 + r2)**-0.5
   C = (1 - 2*r2) / (2 + 4*r2)

   print(f'B {B:.2f}, C {C:.2f}')

   # term n
   gn = lambda n: B * C**n / sp.factorial(n)
   gns = []

   f = 0j * x
   for n in range(N):
      coef = [gn(n) if i == 2*n else 0. for i in range(2*N + 1)]
      gp = hermite_e.hermeval(x, coef) * g
      gns.append(gp)
      f += gp

   T = 0j * x
   Tn = lambda n: B * C**(n - 1) / sp.factorial(n) * \
           (n / (2*n - 0.5) + \
           C * (2*n + 0.5) / ((2*n + 1.5) * (2*n - 0.5)) - \
           C**2 * (2*n + 1) * (2*n + 2) / ((n + 1) * (2*n + 1.5)))
   Tns = []
   T0 = -Tc * B * (2 + 4*C)/3.
   #print(T0, Tn(1))
   T = T0 * g 
   Tns.append(T.copy())
   for n in range(1, N):
      coef = [Tn(n) if i == 2*n else 0. for i in range(2*N + 1)]
      tp = Tc * hermite_e.hermeval(x, coef) * g
      Tns.append(tp)
      T += tp

   plt.ioff()

   fig = plt.figure(3, (4.5, 4.5))
   fig.clf()

   ax1 = fig.add_subplot(211)

   #ax1.plot(lat, G, 'k--', lw = 2., label = 'f')

   for n in range(N):
      ax1.plot(lat, gns[n].real, lw=1., c = '0.8', label = f'n = {n}')
      ax1.plot(lat, gns[n].imag, lw=1., c = '0.8', ls = '--')

   ax1.plot(lat, f.real, 'k', lw = 2., label = 'sum')
   ax1.plot(lat, f.imag, 'k--', lw = 2.)
   ax1.set(ylim = (-0.5, 1.5))
   ax1.set(xlim = (-50, 50))
   ax1.set(title = 'Gaussian Forcing')
   #ax1.legend()

   ax2 = fig.add_subplot(212)
   for n in range(N):
      ax2.plot(lat, Tns[n].real, lw=1., c = '0.8')#, label = f'n = {n}')
      ax2.plot(lat, Tns[n].imag, lw=1., c = '0.8', ls = '--')

   ax2.plot(lat, T.real, 'k', lw = 3., label = 'Full Sln (real)')
   ax2.plot(lat, T.imag, 'k--', lw = 3., label = 'Full Sln (imag)')
   #ax2.set(ylim = (-.25, 1.))
   #ax2.legend()

   #ax2.plot(ds.lat[:], ds.T[:].real, ':',  c = 'C0', lw=2.)
   #ax2.plot(ds.lat[:], ds.T[:].imag, '-.', c = 'C0', lw=2.)

   ax2.plot(Tapp.lat[:], Tapp[:].real, '-', c = 'C1', lw=1., label = 'Approx Sln (real)')
   ax2.plot(Tapp.lat[:], Tapp[:].imag, '--', c = 'C1', lw=1., label = 'Approx Sln (imag)')

   ax2.set(ylim = (-0.15, 0.15))
   ax2.set(xlim = (-50, 50))
   ax2.set(title = 'Temperature response %s' % taulbl)
   ax2.legend(loc = 'best', frameon = False)

   plt.ion()

   plt.show()
   plt.draw()

   if savefig: 
      fn = figpath + f'T_sum_vs_approx_vs_num_{taulbl}.pdf'
      plt.savefig(fn)
      print(f'Figure saved to {fn}.')

# Computes the temperature response to a single Hermite forcing mode Heₙ(x)e^{-x²/4},
# comparing the analytic Tₙ Heₙ(x)e^{-x²/4} solution with the numerical
# finite-difference solution for the same eigenmode.
def plot_T_Fn(n = 0, tau = 0, D = 4.14e3, F = 0.01 / 86400.):
   '''Plots response to forcing by a hermite function of order n. For now I've only looked at T but other quantities are calculated.'''
   ds = solve_T_num(1.1e6, tau, D, F, eig = n)

   if tau == 0.: al = 0.
   else: al = 1 / (tau * 86400.)

   A = (1j * om + al) / (1j * om)

   l2 = np.sqrt(N2/A) * D / (2 * beta)
   l = np.sqrt(l2)

   print(f'A {A:.2f}; l {l:.2f}')

   lat = pyg.regularlat(181)[:]
   y = (a*lat*np.pi/180.)

   x = y / l

   # Forcing: single Hermite polynomial in x multiplied by a gaussian
   coef = [1 if i == n else 0. for i in range(n + 1)]
   g = np.exp(-x**2 / 4.)
   f = F * hermite_e.hermeval(x, coef) * g

   # Response

   # Prefactors
   Vc = -F / (4.*beta*l)
   Wc = 1j * np.sqrt(A / N2) * F / 4.
   #Tc = -1j / (1j*om) * H / R * np.sqrt(N2 / A) * F / 4.
   Tc = -1j / (1j*om + al) * H / R * np.sqrt(A * N2) * F / 4.
   Uc = 1 / (1j*om) * F / 4.

   # Polynomial coefficients
   pp1 = 1/(n + 1.5)
   pm1 = n/(n - 0.5)

   wp2 = 1/(n + 1.5)
   wp0 = (n + 0.5) / ((n - 0.5) * (n + 1.5))
   wm2 = -n * (n - 1) / (n - 0.5)

   up2 = 1/(n + 1.5)**2
   up0 = (2*n**2 + 2*n - 2.5) / ((n + 1.5)**2 * (n - 0.5))
   um2 = -n * (n - 1) / ((n - 0.5) * (n + 1.5))

   # For now only temperature
   Tcoef = np.zeros(n + 3, 'd')
   Tcoef[n + 2] = wp2
   Tcoef[n] = wp0
   if n > 1: Tcoef[n - 2] = wm2

   T = Tc * hermite_e.hermeval(x, Tcoef) * g

   plt.ioff()

   fig = plt.figure(3, (4.5, 4.5))
   fig.clf()

   ax1 = fig.add_subplot(211)

   ax1.plot(lat, f.real, 'k', lw = 2., label = 'sum')
   ax1.plot(lat, f.imag, 'k--', lw = 2., label = 'sum')

   ax1.plot(ds.lat[:], ds.F[:].real, 'r:', lw=2.)
   ax1.plot(ds.lat[:], ds.F[:].imag, 'r-.', lw=2.)
   #ax1.set(ylim = (-0.5, 1.5))
   #ax1.legend()

   ax2 = fig.add_subplot(212)
   ax2.plot(lat, T.real, 'k', lw = 3., label = 'Re')
   ax2.plot(lat, T.imag, 'k--', lw = 3., label = 'Im')
   #ax2.set(ylim = (-.25, 1.))
   #ax2.legend()

   ax2.plot(ds.lat[:], ds.T[:].real, 'r:', lw=2.)
   ax2.plot(ds.lat[:], ds.T[:].imag, 'r-.', lw=2.)

   plt.ion()

   plt.show()
   plt.draw()

# Computes the full analytic temperature field using the Hermite expansion
# T(x) = Σ Tₙ He₂ₙ(x)e^{-x²/4}, with coefficients Tₙ derived from the
# Gaussian–Hermite forcing decomposition.
def solve_T(L=1100e3, tau = 10, D = 2e3, F = 0.15 / 86400., N=100):
   ''' Returns temperature response to arbirary gaussian forcing using eigenfunction expansion.'''
   if tau == 0.: al = 0.
   else: al = 1 / (tau * 86400.)

   A = (1j * om + al) / (1j * om)

   Tc = -1j / (1j*om + al) * H / R * np.sqrt(A * N2) * F / 4.

   l2 = np.sqrt(N2/A) * D / (2 * beta)

   r2 = l2 / L**2

   print(f'A {A:.2f}; r2 {r2:.2f}; l {l:.2f}, Tc {Tc}')

   lat = pyg.regularlat(181)
   y = (a*lat*np.pi/180.)[:]

   x = y / np.sqrt(l2)

   g = np.exp(-x**2 / 4.)

   B = (0.5 + r2)**-0.5
   C = (1 - 2*r2) / (2 + 4*r2)

   Tn = lambda n: B * C**(n - 1) / sp.factorial(n) * \
           (n / (2*n - 0.5) + \
           C * (2*n + 0.5) / ((2*n + 1.5) * (2*n - 0.5)) - \
           C**2 * (2*n + 1) * (2*n + 2) / ((n + 1) * (2*n + 1.5)))

   Tns = np.zeros(2*N - 1, 'complex128')
   Tns[0] = -B * (2 + 4*C)/3.

   for n in range(1, N):
      Tns[2*n] = Tn(n)

   #print(Tns)

   T = Tc * hermite_e.hermeval(x, Tns) * g

   return pyg.Var((lat,), values=T, name = 'T')

# Asymptotic temperature solution, using the expansion
# T ≈ Tc e^{-x²/4}[1 − (r²−½)x²] for small width mismatch r²−½
def solve_T_approx(L, tau = 10, D = 2e3, F = 0.01 / 86400.):
   ''' Returns approximate temperature response to arbirary gaussian forcing using expansion for small x, C.'''
   if tau == 0.: al = 0.
   else: al = 1 / (tau * 86400.)

   A = (1j * om + al) / (1j * om)

   Tc = -1j / (1j*om + al) * H / R * np.sqrt(A * N2) * F / 4.
   l2 = np.sqrt(N2/A) * D / (2 * beta)

   r2 = l2 / L**2
   print(f'r2: {r2}, N2: {N2}, A: {A}, D:{D}')

   lat = pyg.regularlat(181)
   y = (a*lat*np.pi/180.)[:]

   x = y / np.sqrt(l2)

   B = (0.5 + r2)**-0.5
   C = (1 - 2*r2) / (2 + 4*r2)

   print(f'eps: {r2 - 0.5}; C {C}')

   c0 = -4./3 - 20./21 * C + 54. / 77. * C**2
   c2 = 2./3 - 26./21 * C - 129 / 77. * C**2
   c4 = 2./7 * C - 96. / 77 * C**2
   c6 = 1. / 11 * C**2

   print(f'c0-6: {c0}, {c2}, {c4}, {c6}')

   T = Tc * B * (c0 + c2 * x**2 + c4 * x**4 + c6 * x**6) * np.exp(-x**2 / 4.)

   return pyg.Var((lat,), values=T, name = 'T')

# Full numerical solution of the meridional structure ODE: solves
# ψ_{yy} − (y²/4ℓ²)ψ = F using finite differences, computes v, w, T, and u, and
# evaluates u via two independent formulas (streamfunction and thermal-wind) as a consistency check.
def solve_T_num(L=1100e3, tau = 10, D = 2e3, F = 0.15 / 86400., eig = False):
   '''Returns numerical solution to arbitrary gaussian forcing if eig = False.
   If eig is an integer, uses a Hermite function of order eig instead of the gaussian.
   Returns a pygeode data set with F, psi, v, w, t, and two versions of u (that should be identical).'''

   if tau == 0.: al = 0.
   else: 
      al = 1 / (tau * 86400.)
      #print(om/al)
      #print(om, al)

   lat = pyg.regularlat(360)
   Y = (a*lat*np.pi/180.)
   y = Y[:]

   A = 1j * om * N2
   C = -(1j * om + al) * beta**2 / D**2

   idy = 1./(y[1] - y[0])

   # Simple finite difference approximation to differential operator with
   # boundary conditions fixing psi = 0 at the poles.

   im1 =    A*idy**2 + 0 * y[1:-2]
   i0  = -2*A*idy**2 + C * y[1:-1]**2
   ip1 =    A*idy**2 + 0 * y[2:-1]

   Lpsi = sparse.diags([im1, i0, ip1], [-1, 0, 1], format='csc', dtype='complex128')

   if eig is False:
      Fy = F * np.exp(-y**2 / (2*L**2))
   else:
      Afac = (1j * om + al) / (1j * om)
      l2 = np.sqrt(N2/Afac) * D / (2 * beta)
      x = y / np.sqrt(l2)
      Nn = [0] * eig + [1]
      Fy = F * hermite_e.hermeval(x, Nn) * np.exp(-x**2 / 4)

   Rhs = 1j * (1j * om + al) * beta * y / D * Fy

   psi = sparse.linalg.spsolve(Lpsi, Rhs[1:-1])

   psi = np.concatenate([[0.], psi, [0.]])

   F = pyg.Var((lat,), values = Fy, name = 'F')
   Psi = pyg.Var((lat,), values=psi, name = 'psi')

   v = (-1j / D * Psi).rename('V')
   w = Psi.deriv('lat', dx=Y).rename('W')

   T = -N2 / (al + 1j*om) * H / R * w
   T = T.rename('T')

   # Confirm that we can compute u both ways
   u1 = (F + beta * Y * v) / (1j * om)
   u2 = 1j * R / H * D / (beta*Y) * T.deriv('lat', dx=Y)

   u1 = u1.rename('U1')
   u2 = u2.rename('U2')

   return pyg.asdataset([F, Psi, v, w, u1, u2, T])
   #return T


# Plots how the meridional response (u, w, T) changes with radiative damping τ
# by solving the full meridional ODE numerically for several τ values. Shows
# how damping alters the width and amplitude of the wind, vertical velocity, and temperature fields.
def plot_response_vs_alpha(L=1100e3, D = 4.14e3, F = 0.10 / 86400., fig = 1, savefig = False):
   ''' Plots U, T, W, V as a function of latitude for a fixed gaussian forcing structure and
   different values of radiative damping.'''
   #taus = [0., 1000, 300, 100, 30, 10][::-1]
   taus = [0., 100, 40, 10.][::-1]
   lbls = [f'{t} d' for t in taus[:-1]] + ['None']

   dss = []

   for tau in taus:
      ds = solve_T_num(L = L, D = D, F = F, tau = tau)
      dss.append(ds)

   def make_panel(var, label, unit, scale = 1):
      ax = pyg.plot.AxesWrapper(size = (4.8, 2.8))
      #ax = pyg.showlines([ds[var].real() for ds in dss], labels = lbls, size=(5.1, 4.0))

      for i, ds in enumerate(dss):
         pyg.vplot(scale*ds[var].real(), '-',  c = 'C%d' % i, label = lbls[i], axes=ax)
         pyg.vplot(scale*ds[var].imag(), '--', c = 'C%d' % i, label = ''     , axes=ax)

      if var == 'U1':
         # Plot second calculation of U as well (see solve_T_num) to verify solution
         for i, ds in enumerate(dss):
            pyg.vplot(scale*ds['U2'](i_lat=(0, -1, 4)).real(), 'o',  c = 'C%d' % i, label = '', axes=ax)
            pyg.vplot(scale*ds['U2'](i_lat=(0, -1, 4)).imag(), 'x', c = 'C%d' % i, label = '', axes=ax)


      ax.setp(title = f'{label}', \
              xlim = (-70, 70), ylabel = unit)
      return ax

   plt.ioff()

   axU = make_panel('U1', 'Zonal Wind', '[m/s]')
   axT = make_panel('T', 'Temperature', '[K]')
   axW = make_panel('W', 'Vertical Wind', '[mm/s]', 1000.)
   axV = make_panel('V', 'Meridional Wind', '[mm/s]', 1000.)

   axU.text(-60, 10, f'L {int(L*1e-3)} km\nD {D*1e-3} km\nF {F*86400:.2} m/s/d')
   axV.legend(loc = 'lower right', frameon = False, fontsize='small', ncol=2)

   ax = pyg.plot.grid([[axU, axT], [axW, axV]])

   plt.ion()

   ax.render(fig)

   if savefig: 
      fn = figpath + 'response_vs_tau.pdf'
      plt.savefig(fn)
      print(f'Figure saved to {fn}.')

# Plots how the meridional response (u, w, T) varies with the forcing width L
# by solving the full meridional ODE numerically for several L values. Shows
# how the width of the imposed Gaussian forcing alters the structure and amplitude of the response fields.
def plot_response_vs_L(D = 4.14e3, tau = 40, F = 0.10 / 86400., fig=2, savefig = False):
   ''' Plots U, T, W, V as a function of latitude for a gaussian forcing with different meridional
   length scales at a fixed value of radiative damping.'''

   Ls = [110e3 * deg for deg in [5, 8, 10, 12, 15]]
   lbls = [f'{int(l*1e-3)} km' for l in Ls] 

   if tau == 0: taulbl = 'tau: None'
   else: taulbl = f'tau: {tau} d'

   dss = []
   Tsc = []

   for L in Ls:
      print('L %d, T scaling %.2f' % (L, F * H * beta * L**2 / (R * D * om)) )
      Tsc.append(F * H * beta * L**2 / (R * D * om))
      ds = solve_T_num(L = L, D = D, F = F, tau = tau)
      dss.append(ds)

   def make_panel(var, label, unit, scale = 1):
      ax = pyg.plot.AxesWrapper(size = (4.8, 2.8))
      #ax = pyg.showlines([ds[var].real() for ds in dss], labels = lbls, size=(5.1, 4.0))

      for i, ds in enumerate(dss):
         pyg.vplot(scale*ds[var].real(), '-',  c = 'C%d' % i, label = lbls[i], axes=ax)
         pyg.vplot(scale*ds[var].imag(), '--', c = 'C%d' % i, label = ''     , axes=ax)

      ax.setp(title = f'{label}', \
              xlim = (-70, 70), ylabel = unit)
      return ax

   plt.ioff()

   axU = make_panel('U1', 'Zonal Wind', '[m/s]')
   axT = make_panel('T', 'Temperature', '[K]')
   axW = make_panel('W', 'Vertical Wind', '[mm/s]', 1000.)
   axV = make_panel('V', 'Meridional Wind', '[mm/s]', 1000.)

   #for i, Ts in enumerate(Tsc):
      #axT.plot([0], [Ts], 'o', c = 'C%d' % i)

   axU.text(-60, 10, f'{taulbl}\nD {D*1e-3} km\nF {F*86400:.2} m/s/d')
   axV.legend(loc = 'lower right', frameon = False, fontsize='small', ncol=2)

   ax = pyg.plot.grid([[axU, axT], [axW, axV]])

   plt.ion()

   ax.render(fig)

   if savefig: 
      fn = figpath + 'response_vs_L_tau%d.pdf' % tau
      plt.savefig(fn)
      print(f'Figure saved to {fn}.')

# Compares numerical equatorial T and W amplitudes with simple analytic scaling
# laws by sweeping the radiative damping time τ. Uses the full numerical ODE
# solution for each τ and plots numerical vs. scaled values and their ratios.
def plot_scaling_vs_tau(L = 1100e3, D = 4.14e3, F = 0.10 / 86400., fig=3, savefig=False):
   '''Plots comparison of T and W against scaling relationships derived
   from assuming a fixed meridional lengthscale against radiative damping timescale.'''

   als = np.linspace(0, 0.1, 181)
   taus = 1. / als
   taus[0] = 0.

   #als = np.array([1/(tau * 86400) if tau > 0 else 0. for tau in taus])
   
   sc_Ts = []
   ca_Ts = []

   sc_Ws = []
   ca_Ws = []

   for al, tau in zip(als, taus):
      ds = solve_T_num(L = L, D = D, F = F, tau = tau)
      ca_Ts.append(ds.T(m_lat = (-5, 5))[()])
      ca_Ws.append(ds.W(m_lat = (-5, 5))[()])

      if al/86400. < om: fac = 1.
      else: fac = om * tau * 86400.

      #print(tau, fac)
      sc_Ts.append(F * H * beta * L**2 / (R * D * om))
      sc_Ws.append(F * beta * L**2 / (fac * N2 * D))

   ca_Ts = np.abs(np.array(ca_Ts))
   sc_Ts = np.abs(np.array(sc_Ts))

   ca_Ws = np.abs(np.array(ca_Ws))
   sc_Ws = np.abs(np.array(sc_Ws))

   plt.ioff()

   f = plt.figure(fig, (6.3, 2.5))
   f.clf()

   ax1 = f.add_subplot(121)

   ax1.plot(als, ca_Ts, '-', label = 'Calculated')
   ax1.plot(als, sc_Ts, '--', label = 'Scaling')
   ax1.plot(als, sc_Ts/ca_Ts, '--', label = 'Ratio')
   ax1.plot(als, sc_Ws/ca_Ws, '-.', label = 'Ratio (W)')

   ax1.set_xscale('log')
   ax1.set_title('Temperature')
   ax1.set_ylabel('K')
   ax1.set_xlabel(r'd$^{-1}$')
   ax1.legend(loc='best', frameon=False)

   ax2 = f.add_subplot(122)

   ax2.plot(als, 1e3*ca_Ws, '-', label = 'Calculated')
   ax2.plot(als, 1e3*sc_Ws, '--', label = 'Scaling')
   ax2.axvline(x = 1./40., c = 'k', lw = 1.)
   #ax2.plotal * 86400.(Ls*1e-3, sc_Ws/ca_Ws, '--', label = 'Ratio')

   ax2.text(0.1, 0.6, 'L = %d' % (L*1e-3), transform=ax2.transAxes)

   ax2.set_xscale('log')
   ax2.set_title('Vertical Velocity')
   ax2.set_ylabel('mm/s')
   ax2.set_xlabel(r'd$^{-1}$')

   #ax.plot(Ls*1e-3, ca_Ts, '-', lbl = 'Calculated')
   #ax.plot(Ls*1e-3, sc_Ts, '--', lbl = 'Scaling')
   plt.tight_layout()

   plt.ion()

   plt.show()
   plt.draw()

   if savefig: 
      fn = figpath + 'scaling_comparison_vs_alpha_L%d.pdf' % L
      plt.savefig(fn)
      print(f'Figure saved to {fn}.')

# Compares numerical equatorial T and W amplitudes with simple L²-based scaling
# laws by solving the full meridional ODE numerically for several forcing widths L.
# Plots numerical vs. scaled amplitudes and their ratios to assess scaling accuracy.
def plot_scaling_vs_L(D = 4.14e3, tau = 40, F = 0.10 / 86400., fig=4, savefig=False):
   '''Plots comparison of T and W against scaling relationships derived
   from assuming a fixed meridional lengthscale against forcing meridional lengthscale.'''

   Ls = np.array([110e3 * deg for deg in np.arange(2, 15, 0.2)])
   #Ls = np.array([110e3 * deg for deg in [10.]])
   
   if tau == 0: 
      taulbl = 'tau: None'
      fac = 1.
   else: 
      taulbl = f'tau: {tau} d'
      fac = tau * 86400 * om

   sc_Ts = []
   ca_Ts = []

   sc_Ws = []
   ca_Ws = []

   for L in Ls:
      ds = solve_T_num(L = L, D = D, F = F, tau = tau)
      ca_Ts.append(pyg.absolute(ds.T(m_lat = (-5, 5)))[()])
      ca_Ws.append(pyg.absolute(ds.W(m_lat = (-5, 5)))[()])

      sc_Ts.append(F * H * beta * L**2 / (R * D * om))
      sc_Ws.append(F * beta * L**2 / (fac * N2 * D))

   ca_Ts = np.array(ca_Ts)
   sc_Ts = np.array(sc_Ts)

   ca_Ws = np.array(ca_Ws)
   sc_Ws = np.array(sc_Ws)

   plt.ioff()

   f = plt.figure(fig, (6.3, 2.5))
   f.clf()

   ax1 = f.add_subplot(121)

   ax1.plot(Ls*1e-3, ca_Ts, '-', label = 'Calculated')
   ax1.plot(Ls*1e-3, sc_Ts, '--', label = 'Scaling')
   ax1.plot(Ls*1e-3, sc_Ts/ca_Ts, '--', label = 'Ratio')
   #ax1.plot(Ls*1e-3, sc_Ws/ca_Ws, '--', label = 'Ratio (W)')

   ax1.set_title('Temperature')
   ax1.set_ylabel('K')
   ax1.set_xlabel('km')
   ax1.legend(loc='best', frameon=False)

   ax2 = f.add_subplot(122)

   ax2.plot(Ls*1e-3, 1e3*ca_Ws, '-', label = 'Calculated')
   ax2.plot(Ls*1e-3, 1e3*sc_Ws, '--', label = 'Scaling')
   #ax2.plot(Ls*1e-3, sc_Ws/ca_Ws, '--', label = 'Ratio')

   ax2.text(0.1, 0.6, taulbl, transform=ax2.transAxes)

   ax2.set_title('Vertical Velocity')
   ax2.set_ylabel('mm/s')
   ax2.set_xlabel('km')

   #ax.plot(Ls*1e-3, ca_Ts, '-', lbl = 'Calculated')
   #ax.plot(Ls*1e-3, sc_Ts, '--', lbl = 'Scaling')
   plt.tight_layout()

   plt.ion()

   plt.show()
   plt.draw()

   if savefig: 
      fn = figpath + 'scaling_comparison_vs_L_tau%d.pdf' % tau
      plt.savefig(fn)
      print(f'Figure saved to {fn}.')

# Plots the mismatch parameter r² = ℓ²/L² as a function of the forcing width L,
# using a fixed intrinsic scale ℓ from the chosen τ. Helps visualise how forcing
# width controls Hermite-mode excitation and the validity of analytic approximations.
def plot_r2(D = 4.14e3, F = 0.10 / 86400., fig=4, savefig = False):
   ''' Plots r2 parameter against forcing meridional length scale. '''

   Ls = np.array([110e3 * deg for deg in np.arange(8, 14, 1.)])

   def add_r2(ax, tau, lbl):
      if tau == 0: 
         taulbl = r'$\tau$: None'
         al = 0.
      else: 
         taulbl = r'$\tau$'f': {tau} d'
         al = 1 / (tau * 86400.)

      A = (1j * om + al)/(1j * om)
      #Ao3 = -(1j * 0.2) / (0.2  * 0.5)
      l2 = np.sqrt(N2/A) * D / (2 * beta)

      r2 = l2 / Ls**2

      ax.plot(r2.real, r2.imag, 'o', label = taulbl)

      if lbl:
         ax.text(r2[0].real, r2[0].imag - 0.05, r'8$^\circ$', ha='center', va = 'top')
         ax.text(r2[-1].real, r2[-1].imag - 0.05, r'13$^\circ$', ha='center', va = 'top')


   plt.ioff()

   f = plt.figure(fig, (4.8, 4.3))
   f.clf()

   #ax1 = f.add_subplot(111)

   #ax1.plot(Ls*1e-3, r2.real, '-', label = 'real')
   #ax1.plot(Ls*1e-3, r2.imag, '--', label = 'imag')

   #ax1.axhline(y = 0.5, c = 'k')

   #ax1.set_title(taulbl)
   #ax1.set_ylim(-2, 2)
   #ax1.set_ylabel(r'$r^2$')
   #ax1.set_xlabel('L')
   #ax1.legend(loc='best', frameon=False)

   ax1 = f.add_subplot(111, aspect='equal')

   add_r2(ax1, 10, False)
   add_r2(ax1, 40, True)
   add_r2(ax1, 0,  False)

   ax1.plot([0.5], [0.0], 'kx')

   ax1.set_title(r'$r^2 = \epsilon + 0.5$')
   ax1.set_ylim(-0.1, 0.9)
   ax1.set_xlim(0.1, 1.1)
   ax1.set_xlabel('Real')
   ax1.set_ylabel('Imaginary')
   ax1.legend(loc='best', frameon=False)

   plt.tight_layout()

   plt.ion()

   plt.show()
   plt.draw()

   if savefig: 
      fn = figpath + 'r2_vs_L_tau.pdf'
      plt.savefig(fn)
      print(f'Figure saved to {fn}.')


# Plots the Hermite forcing parameter C = (1−2r²)/(2+4r²) as a function of the
# forcing width L, using the intrinsic scale ℓ set by τ. Shows how C varies with L
# and how strongly higher Hermite modes are excited for different forcing widths.
def plot_C(D = 4.14e3, F = 0.10 / 86400., fig=4, savefig = False):
   ''' Plots r2 parameter against forcing meridional length scale. '''

   Ls = np.array([110e3 * deg for deg in np.arange(8, 14, 1.)])

   def add_C(ax, tau, lbl):
      if tau == 0: 
         taulbl = r'$\tau$: None'
         al = 0.
      else: 
         taulbl = r'$\tau$'f': {tau} d'
         al = 1 / (tau * 86400.)

      A = (1j * om + al)/(1j * om)
      #Ao3 = -(1j * 0.2) / (0.2  * 0.5)
      l2 = np.sqrt(N2/A) * D / (2 * beta)

      r2 = l2 / Ls**2
      C = (1 - 2*r2) / (2 + 4*r2)

      ax.plot(C.real, C.imag, 'o', label = taulbl)

      if lbl:
         ax.text(C[0].real, C[0].imag + 0.03, r'8$^\circ$', ha='center', va = 'center')
         ax.text(C[-1].real, C[-1].imag + 0.03, r'13$^\circ$', ha='center', va = 'center')


   plt.ioff()

   f = plt.figure(fig, (4.8, 4.3))
   f.clf()

   #ax1 = f.add_subplot(111)

   #ax1.plot(Ls*1e-3, r2.real, '-', label = 'real')
   #ax1.plot(Ls*1e-3, r2.imag, '--', label = 'imag')

   #ax1.axhline(y = 0.5, c = 'k')

   #ax1.set_title(taulbl)
   #ax1.set_ylim(-2, 2)
   #ax1.set_ylabel(r'$r^2$')
   #ax1.set_xlabel('L')
   #ax1.legend(loc='best', frameon=False)

   ax1 = f.add_subplot(111, aspect='equal')

   add_C(ax1, 10, False)
   add_C(ax1, 40, True)
   add_C(ax1, 0,  False)

   ax1.plot([0.0], [0.0], 'kx')

   ax1.set_title(r'$C = (1 - 2 r^2)/(2 + 4r^2)$')
   ax1.set_ylim(-0.35, 0.35)
   ax1.set_xlim(-0.35, 0.35)
   ax1.set_xlabel('Real')
   ax1.set_ylabel('Imaginary')
   ax1.legend(loc='best', frameon=False)

   plt.tight_layout()

   plt.ion()

   plt.show()
   plt.draw()

   if savefig: 
      fn = figpath + 'C_vs_L_tau.pdf'
      plt.savefig(fn)
      print(f'Figure saved to {fn}.')
