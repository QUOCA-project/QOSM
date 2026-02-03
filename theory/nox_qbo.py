import numpy as np
import pygeode as pyg
from matplotlib import pyplot as plt
import model as mdl

from an_fits_ref import *

Z = pyg.Height(np.linspace(12e3, 40e3, 101))

H = 7000.
#z = -H*pyg.log(pre/1000.)
pre = pyg.Pres(1000 * np.exp(-Z[:] / H))

ma = 0.02895997  #molar mass of air kg mol-1
mo3 = 0.047997   #molar mass of ozone kg mol-1
mnox = 0.046     #molar mass of nox kg mol-1 (NOT ACCURATE)

datapath = 'input/'

#rce_fns = {'ref' : 'rce_pce_lower_dresw_with_dnox_updated.nc'}
rce_fns = {'ref' : 'background_upwelling.nc'}

# Time slice to use from RCE runs for comparison
interval = (4*840., 5*840.)

def format_rad(den=2):
# {{{
   def fmt(val, pos = None, den=den):
      num = int(np.round(val * den))
      gcd = np.gcd(den, num)

      num = num // gcd
      den = den // gcd
      if num == 0:
         return r'0'
      elif den > 1:
         if num == 1:
            return r'$\frac{\pi}{%d}$' % den
         elif num == -1:
            return r'$\frac{-\pi}{%d}$' % den
         else:
            return r'$\frac{%d \pi}{%d}$' % (num, den)
      else:
         if num == 1:
            return r'$\pi$'
         elif num == -1:
            return r'$-\pi$'
         else:
            return r'$%d \pi$' % num
   return fmt
# }}}

def set_rad_axis(ax, den = 2):
   ax.setp_xaxis(major_locator = plt.MultipleLocator(1 / den), major_formatter = plt.FuncFormatter(format_rad(den)))

def open_pop_file(run = None):
# {{{ 
   if run == None: run = 'ref'

   fn = 'era5_pops_regressed_onto_u_pop_4_200hPa_5NS.nc'

   dpops = pyg.open(datapath + fn)

   #time = pyg.NamedAxis(dpops.phase[:] * 840. / (2 * np.pi), 'time')
   #time = pyg.modeltime360n(interval[0], n = len(dpops.phase), step = np.diff(dpops.phase[:])[0] * 840. / (2 * np.pi))
   time = pyg.ModelTime360(np.linspace(interval[0], interval[1], len(dpops.phase)), 
                           startdate = dict(year=2000, month=1, day=1), 
                           units='days')
   dsp = dpops.replace_axes(phase = time).squeeze()

   # Convert ozone from mass mixing ratio to volume mixing ratio
   o3 = dsp.o3_pop1 / mo3 * ma
   dsp = dsp.replace_vars(o3_pop1 = o3.rename('o3_pop1'))

   z_axis = -7 * pyg.log(dsp.pres / 1000.)

   if run in ['ref', 'hlf', 'dbl']:
      msk = (1-0.2*(pyg.tanh((z_axis-20))-pyg.tanh((z_axis-28))))
   elif run in ['dwr']:
      msk = (1-0.2*(pyg.tanh((z_axis-20))+1)) * (-0.5*pyg.tanh(5*(z_axis-25.7)) + 0.5)
   else:
      msk = None

   if msk is not None:
      w = (dsp.resw_pop1 * msk).rename('resw_pop1')
      dsp = dsp.replace_vars(resw_pop1 = w.rename('resw_pop1'))

   return dsp

# }}}

def open_rce_file(run = None):
# {{{
   if run == None: run = 'ref'

   pth = datapath + rce_fns[run]
   ds = pyg.openall(pth)

   time_axis = pyg.ModelTime360(values=ds.time[:] - 152., units = 'days', startdate=dict(year=2000, month=1, day=1))
   pres_axis = pyg.Pres(ds.lyP(i_time=0).squeeze()[:]/1e2)

   return ds.replace_axes(time = time_axis, ly = pres_axis)
# }}}

def open_dnox_file(ref = 'ref'):
# {{{
   dsr = open_rce_file(ref)

   fn_nox = datapath + 'dnox.nc'
   dnox = pyg.open(fn_nox).replace_axes(ly = dsr.pres)
   time_axis = pyg.ModelTime360(values=dnox.time[:] + 4*840, units = 'days', startdate=dict(year=2000, month=1, day=1))
   return dnox.replace_axes(time=time_axis)
# }}}

def get_S_dChidz(run = None, recalc = False):
# {{{
   if run == None: run = 'ref'

   Sfn = datapath + 'SdO3dz_%s.nc' % run
   #if recalc or not os.path.exists(Sfn):
      #save_S_dChidz(run)

   ds = pyg.open(Sfn)
   return ds.Sm, ds.Sd, ds.Om, ds.Od, ds.rat, ds.inv, ds.reg, ds.ireg
# }}}

def gamma_n2o(zs):
# {{{
   lt = 6 + 5 * pyg.exp(-(zs - 20e3)/6e3)
   gamma = (10**-lt).rename('gamma_N2O')
   return gamma
# }}}

def upwelling(zs):
# {{{
   # Increase  from .3mm/s at 20 km to .5mm/s at 35 km
   w = 0*zs + 0.0005 - (35 - zs*1e-3) / (35 - 20) * 0.0002 * (zs*1e-3 < 35.) 
   return w.rename('w')
# }}}

def qbo_upwelling(zs):
# {{{
   # Increase  from .0mm/s at 18 km to .2mm/s at 35 km; back to zero at 50 km
   a0 = 0.0
   z0 = 18e3

   a1 = 0.0002
   z1 = 35e3

   a2 = 0.
   z2 = 40e3

   zlmsk = (zs > z0) * (zs < z1)
   zumsk = (zs >= z1) * (zs < z2)

   phs = pyg.NamedAxis(np.linspace(0, 2*np.pi, 101), 'phase')

   amp = (zs - z0) / (z1 - z0) * a1 * zlmsk \
       + (z2 - zs) / (z2 - z1) * a1 * zumsk

   p0 = -(zs - z0) / (z1 - z0) * 2 * np.pi
   #p0 = pyg.clip(p0, 0, 2*np.pi)

   wp = pyg.cos(phs - p0) * amp
   return wp.rename("w'")
# }}}

def fit_amp_phase(v, plot = False, off = 0.):
# {{{
   if v.hasaxis('time'):
      phs = (2 * np.pi * (v.time - interval[0]) / 840.)
      ax = 'time'
   else:
      phs = v.phase
      ax = 'phase'

   if v.hasaxis('z'):
      axz = v.z
      ref_height = dict(z = 30e3)
   else:
      axz = v.pres
      ref_height = dict(pres = 30)

   ca = (2*pyg.cos(phs) * v).mean(ax)
   sa = (2*pyg.sin(phs) * v).mean(ax)

   amp = pyg.sqrt(ca**2 + sa**2)

   phase = pyg.arctan2(sa, ca)
   df = (np.pi + phase.diff()) % (2 * np.pi) - np.pi
   p0 = phase.slice[:1]
   p1 = df.cumsum(0, v0 = phs[:][1]).replace_axes(**{axz.name:axz.slice[1:]})

   phs_c = pyg.concatenate([p0, p1])

   p30 = phs_c(**ref_height)[:][0]
   off -= p30 - (p30 % (2 * np.pi))
   phs_c += off

   ds = pyg.asdataset([amp.rename('amp'), 
                       phs_c.rename('phase'),
                       ca.rename('re'),
                       sa.rename('im')])

   if plot:
      w = to_timeseries(ds, phs, 'w')
      pyg.showgrid([v, w], ncol=1, fig=0)

   return ds
# }}}

def to_timeseries(ds, phase, name = 'v'):
   w = ds.amp * np.cos(phase - ds.phase)
   return w.rename(name)

def to_complex(ds, name = 'v'):
# {{{
   v = ds.amp * pyg.exp(1j * ds.phase)
   return v.rename(name)
# }}}

def to_amp_phase(v, off = 0.):
# {{{
   amp = pyg.absolute(v).rename('amp')
   phs = pyg.angle(v).rename('phase')
   df = (np.pi + phs.diff()) % (2 * np.pi) - np.pi
   p0 = phs.slice[:1]
   p1 = df.cumsum(0, v0 = phs[:][1]).replace_axes(pres=phs.pres.slice[1:])

   phs_c = pyg.concatenate([p0, p1])

   p30 = phs_c(pres = 80)[:][0]
   off -= p30 - (p30 % (2 * np.pi))
   phs_c += off

   re  = pyg.real(v).rename('re')
   im  = pyg.imag(v).rename('im')

   return pyg.asdataset([amp, phs_c.rename('phase'), re, im])
# }}}

def plot_gamma():
# {{{
   gamma = gamma_n2o(Z)

   plt.ioff()
   ax = pyg.showvar(np.log10(gamma))
   #ax.setp(xscale = 'log')
   ax.setp_xaxis(major_formatter=plt.FormatStrFormatter(r'10$^{%d}$'))
   ax.setp(xlabel = r'Damping rate (s$^{-1}$)', title = r'$\gamma_{N_2O}$')

   plt.ion()
   ax.render(1)
# }}}

def plot_w():
# {{{
   run = 'ref'
   dsr = open_rce_file(run)
   dsp = open_pop_file(run)

   W0 = dsr.strat_up_ozone / 1e3
   Wp = fit_amp_phase(dsp.resw_pop1)

   w = upwelling(Z).replace_axes(z = pre)
   qw = fit_amp_phase(qbo_upwelling(Z)).replace_axes(z = pre)

   def phs(d, offset): return (d.phase - offset) / np.pi

   plim = (150., 4.8)

   plt.ioff()

   axw0 = pyg.showlines([1e3*w, 1e3*W0], labels = ['Simple Fit', 'ERA 5'])
   axwp = pyg.showlines([1e3*qw.amp, 1e3*Wp.amp], labels = ['Simple Fit', 'Modified ERA 5'])
   axwP = pyg.showlines([phs(qw, 0), 
                         phs(Wp, 0)], labels = ['Simple Fit', 'Modified ERA 5'])

   axw0.setp(xlabel = 'mm/s', title = r'Background upwelling w$^\asterisk_0$', ylim = plim, xlim = (0, 0.7))
   axwp.setp(xlabel = 'mm/s', title = r'QBO upwelling (amplitude) w$^\asterisk$', ylim = plim)
   axwP.setp(title = r'QBO upwelling (phase) w$^\asterisk$', ylim = plim)
   set_rad_axis(axwP)

   ax = pyg.plot.grid([[axw0], [axwp], [axwP]])

   plt.ion()
   ax.render(2)
# }}}

def plot_LN2O():
# {{{
   # Predict vertical structure in N2O
   gamma = gamma_n2o(Z)
   w = upwelling(Z)
   qw= qbo_upwelling(Z)

   def get_N2O(w, g):
      L = w/g
      T = (1/L).integrate('z', dx = Z)

      N = 270*pyg.exp(-T)
      return L, T, N

   Lb, Tb, Nb = get_N2O(w, gamma)
   L1, T1, N1 = get_N2O(w + qw(s_phase = 0),     gamma)
   L2, T2, N2 = get_N2O(w + qw(s_phase = np.pi), gamma)

   plt.ioff()

   ax0 = pyg.showlines([L*1e-3 for L in [Lb, L1, L2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax0.setp(xlabel = 'km', title = r'L$_{N_2O}$', xscale='log', ylim = (20e3, 40e3), xlim = (0.5, 10000.))
   ax0.setp_xaxis(major_locator = plt.LogLocator())

   ax1 = pyg.showlines([T for T in [Tb, T1, T2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax1.setp(xlabel = 'decay scales', title = r'T$_{N_2O}$', ylim = (20e3, 40e3), xlim = (-1, 5))
   #ax1.setp_xaxis(major_locator = plt.LogLocator())

   ax2 = pyg.showlines([N for N in [Nb, N1, N2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax2.setp(xlabel = r'ppbv', title = r'N$_2$O', ylim = (20e3, 40e3), xscale = 'log')
   ax2.setp_xaxis(major_locator = plt.LogLocator())

   ax = pyg.plot.grid([[ax0, ax1, ax2]])

   plt.ion()
   ax.render(3)
# }}}

def plot_ndcoefs():
# {{{
   omega = 2 * np.pi / (30*28*86400.)
   gamma = gamma_n2o(Z).replace_axes(height = pre)

   plt.ioff()
   ax = pyg.showvar(omega / gamma)
   ax.setp(xscale = 'log')
   #ax.setp_xaxis(major_formatter=plt.FormatStrFormatter(r'10$^{%d}$'))
   ax.setp_xaxis(major_locator=plt.LogLocator())#, maplt.FormatStrFormatter(r'10$^{%d}$'))
   ax.setp(xlabel = r'$\omega / \gamma_{N_2O}$', title = r'$\omega / \gamma_{N_2O}$', xlim = (0.1, 50.))

   rat = 1 / (1 + 1j * omega / gamma)
   ax1 = pyg.showvar(pyg.absolute(rat))
   #ax1.setp(xscale = 'log')
   #ax1.setp_xaxis(major_locator=plt.LogLocator())#, maplt.FormatStrFormatter(r'10$^{%d}$'))
   ax1.setp(xlabel = '', title = r'$1 / (1 + i\omega / \gamma_{N_2O})$')

   plt.ion()
   ax.render(1)

# }}}

def validate_model_adv():
# {{{
   plim = (150., 4.8)

   zb = -H * np.log(plim[0] / 1000.)
   zt = -H * np.log(plim[1] / 1000.)

   St = mdl.BaseState(zb, zt, Nz = 801)

   st_pres = pyg.Pres(St.ps)
   st_zs = pyg.Height(St.zs)

   St.wp[:] = 0.
   St.w0[:] = 0.0003

   St.T0   = 1.
   St.O30  = 2.
   St.N2O0 = 3.
   St.NOx0 = 4.

   # Turn off any interactions
   St.aT[:]    = 0.01 / 86400.
   St.aO3[:]   = 0.
   St.dT[:]    = 0.
   St.dO3[:]   = 0.02 / 86400.
   St.dNOx[:]  = 0.
   St.gN2O[:]  = 0.03 / 86400. 
   St.gNOx[:]  = 0.
   St.eN2O[:]  = 0.
   St.eNOx[:]  = 0.04 / 86400. 

   T, O3, N2O, NOx = St.solve()

   T   = pyg.Var((st_zs, ), name = 'T',   values = T)
   O3  = pyg.Var((st_zs, ), name = 'O3',  values = O3)
   N2O = pyg.Var((st_zs, ), name = 'N2O', values = N2O)
   NOx = pyg.Var((st_zs, ), name = 'NOx', values = NOx)

   Ta    = St.T0   * pyg.exp(-((1j * St.omega + St.aT[0])   / St.w0[0]) * (st_zs - zb))
   Ta    = St.T0   * pyg.exp(-((1j * St.omega + St.aT[0] + St.w0[0] * mdl.R / (mdl.cp * mdl.H)) / St.w0[0]) * (st_zs - zb))
   O3a   = St.O30  * pyg.exp(-((1j * St.omega + St.dO3[0])  / St.w0[0]) * (st_zs - zb))
   N2Oa  = St.N2O0 * pyg.exp(-((1j * St.omega + St.gN2O[0]) / St.w0[0]) * (st_zs - zb))
   NOxa  = St.NOx0 * pyg.exp(-((1j * St.omega + St.eNOx[0]) / St.w0[0]) * (st_zs - zb))

   plt.ioff()
   axs = []
   for i, (v, va) in enumerate(zip([T, O3, N2O, NOx], [Ta, O3a, N2Oa, NOxa])):
      ax = pyg.plot.AxesWrapper(size=(2.5, 3))
      pyg.vplot(va.real(), c = 'k',     ls = '-',  lw = 2., axes = ax)
      pyg.vplot(va.imag(), c = 'k',     ls = '--', lw = 2., axes = ax)
      pyg.vplot(v.real(),  c = f'C{i}', ls = '-',  lw = 2., axes = ax)
      pyg.vplot(v.imag(),  c = f'C{i}', ls = '--', lw = 2., axes = ax)
      ax.axvline(x = 0, c = 'k', lw = 1.)
      axs.append(ax)

   ax = pyg.plot.grid([axs])
   plt.ion()

   ax.render(1)
# }}}

def validate_model_nox():
# {{{
   plim = (150., 4.8)

   zb = -H * np.log(plim[0] / 1000.)
   zt = -H * np.log(plim[1] / 1000.)

   St = mdl.BaseState(zb, zt, Nz = 801)

   st_pres = pyg.Pres(St.ps)
   st_zs = pyg.Height(St.zs)

   St.wp[:] = 0.
   St.w0[:] = 0.0003

   St.omega = 0.0

   St.T0   = 1.
   St.O30  = 2.
   St.N2O0 = 0.1
   St.NOx0 = 0.

   # Turn off any interactions
   St.aT[:]    = 0.01 / 86400.
   St.aO3[:]   = 0.
   St.dT[:]    = 0.
   St.dO3[:]   = 0.02 / 86400.
   St.dNOx[:]  = 0.
   St.gN2O[:]  = 0.0 / 86400. 
   St.gNOx[:]  = 0.
   St.eN2O[:]  = 0.1 / 86400.
   St.eNOx[:]  = 0.0 / 86400. 

   T, O3, N2O, NOx = St.solve()

   T   = pyg.Var((st_zs, ), name = 'T',   values = T)
   O3  = pyg.Var((st_zs, ), name = 'O3',  values = O3)
   N2O = pyg.Var((st_zs, ), name = 'N2O', values = N2O)
   NOx = pyg.Var((st_zs, ), name = 'NOx', values = NOx)

   Ta    = St.T0   * pyg.exp(-((1j * St.omega + St.aT[0] + St.w0[0] * mdl.R / (mdl.cp * mdl.H)) / St.w0[0]) * (st_zs - zb))
   O3a   = St.O30  * pyg.exp(-((1j * St.omega + St.dO3[0])  / St.w0[0]) * (st_zs - zb))
   N2Oa  = St.N2O0 * pyg.exp(-((1j * St.omega + St.gN2O[0]) / St.w0[0]) * (st_zs - zb))
   NOxa  = St.NOx0 + (St.eN2O[0] / St.w0[0]) * St.N2O0 * (st_zs - zb)

   plt.ioff()
   axs = []
   for i, (v, va) in enumerate(zip([T, O3, N2O, NOx], [Ta, O3a, N2Oa, NOxa])):
      ax = pyg.plot.AxesWrapper(size=(2.5, 3))
      pyg.vplot(va.real(), c = 'k',     ls = '-',  lw = 2., axes = ax)
      pyg.vplot(va.imag(), c = 'k',     ls = '-',  lw = 1., axes = ax)
      pyg.vplot(v.real(),  c = f'C{i}', ls = '--',  lw = 2., axes = ax)
      pyg.vplot(v.imag(),  c = f'C{i}', ls = '--', lw = 1., axes = ax)
      ax.axvline(x = 0, c = 'k', lw = 1.)
      axs.append(ax)

   ax = pyg.plot.grid([axs])
   plt.ion()

   ax.render(1)
# }}}

def run_nox_basestate():
# {{{
   # Get some basic state profiles from data files generated by Alison
   run = 'ref'
   dsr = open_rce_file(run)
   dnx = open_dnox_file(run)

   # Stratification, background ozone profile
   #Sm, Sd, Om, Od, rat, inv, reg, ireg = get_S_dChidz(run)

   # Background upwelling
   W0 = dsr.strat_up_ozone / 1e3

   # Initialize basic state
   plim = (150., 4.8)

   zb = -H * np.log(plim[0] / 1000.)
   zt = -H * np.log(plim[1] / 1000.)

   St = mdl.BaseState(zb, zt, Nz = 301)

   st_pres = pyg.Pres(St.ps)
   st_zs = pyg.Height(St.zs)
   st_zs.plotatts['scalefactor'] = 1e-3
   st_zs.units = 'km'

   St.wp[:] = 0.0
   St.omega = 0.

   #St.w0[:] = 0.0003
   #St.w0[:] = upwelling(st_zs)[:]
   St.w0[:] = W0.interpolate('pres', st_pres)[:]

   St.S0[:]    = 0.
   St.dO3dz[:] = 0.

   St.dN2Odz[:] = 0.
   St.dNOxdz[:] = 0.

   St.T0   = 0.
   St.O30  = 0.
   St.N2O0 = 270e-9  # 270 ppbv
   St.NOx0 = 0.

   St.aT[:]    = a_T (St.ps)  / 86400.
   St.aO3[:]   = a_O3(St.ps)  / 86400.
   St.dT[:]    = d_T (St.ps)  / 86400.
   St.dO3[:]   = d_O3(St.ps)  / 86400.
   St.dNOx[:]  = d_NOx(St.ps) / 86400.
   St.gN2O[:]  = gamma_n2o(st_zs)[:]
   St.gNOx[:]  = 0.
   St.eN2O[:]  = eps_N2O(St.zs)
   #St.eN2O[:]  = 5e-9
   #St.eNOx[:]  = 1.e-7
   St.eNOx[:]  = init_eps_NOy(St.zs)

   T, O3, N2O, NOx = St.solve()

   T   = pyg.Var((st_pres, ), name = 'T',   values = T)
   O3  = pyg.Var((st_pres, ), name = 'O3',  values = O3)
   N2O = pyg.Var((st_pres, ), name = 'N2O', values = N2O)
   NOx = pyg.Var((st_pres, ), name = 'NOx', values = NOx)

   return St, pyg.asdataset([T, O3, N2O, NOx])
# }}}

def run_nox_qbo(config = 'nox_analytical', n2o_source = True):
# {{{
   # Get some basic state profiles from data files generated by Alison
   run = 'ref'
   dsr = open_rce_file(run)
   dsp = open_pop_file(run)
   dnx = open_dnox_file(run)

   # Stratification, background ozone profile
   Sm, Sd, Om, Od, rat, inv, reg, ireg = get_S_dChidz(run)

   # Background upwelling
   W0 = dsr.strat_up_ozone / 1e3

   # QBO Upwelling
   daW = fit_amp_phase(dsp.resw_pop1*1e3)
   Wp = to_complex(daW, 'W') / 1e3

   # Initialize basic state
   plim = (150., 4.8)

   zb = -H * np.log(plim[0] / 1000.)
   zt = -H * np.log(plim[1] / 1000.)

   St = mdl.BaseState(zb, zt, Nz = 301)

   st_pres = pyg.Pres(St.ps)
   st_zs = pyg.Height(St.zs)
   st_zs.plotatts['scalefactor'] = 1e-3
   st_zs.units = 'km'

   #St.wp[:] = 0.
   #St.wp[:] = to_complex(fit_amp_phase(qbo_upwelling(st_zs)))[:]
   St.wp[:] = Wp.real().interpolate('pres', st_pres)[:] + 1j * Wp.imag().interpolate('pres', st_pres)[:]

   St.w0[:] = 0.
   #St.w0[:] = 0.0003
   #St.w0[:] = upwelling(st_zs)[:]
   #St.w0[:] = W0.interpolate('pres', st_pres)[:]

   #St.S0[:]    = 12e-3
   St.S0[:]    = Sm.interpolate('pres', st_pres)[:]
   #St.dO3dz[:] = 5e-4
   St.dO3dz[:] = Om.interpolate('pres', st_pres)[:]

   if config == 'nox_analytical':
      St0, ds0 = run_nox_basestate()
      st_zs = pyg.Height(St0.zs)

      St.dN2Odz[:] = ds0.N2O.deriv('pres', dx=st_zs)[:].real
      St.dNOxdz[:] = ds0.NOx.deriv('pres', dx=st_zs)[:].real
   elif config == 'nox':
      #St.dN2Odz[:] = 1e-11
      St.dN2Odz[:] = init_dN2O_0_dz(St.zs)
      #St.dNOxdz[:] = 1e-12
      #St.dNOxdz[:] = 0.
      St.dNOxdz[:] = init_dNOx_0_dz(St.zs)
   elif config == 'noy':
      # In this case the 'NOx' tracer is treated as NOy by multiplying 
      # observed NOx by a fixed ratio
      r = init_NOx_per_NOy(St.zs)
      noy = init_NOx_0(St.zs) / r
      St.dNOxdz[:] = np.gradient(noy, St.zs)
      #St.dNOxdz[:] = 0.
      St.dN2Odz[:] = init_dN2O_0_dz(St.zs)
   else:
      raise ValueError(f'Unrecognized configuration {config}')

   St.T0   = 0.
   St.O30  = 0.
   St.N2O0 = 0.
   St.NOx0 = 0.

   St.aT[:]    = a_T (St.ps)  / 86400.
   St.aO3[:]   = a_O3(St.ps)  / 86400.
   St.dT[:]    = d_T (St.ps)  / 86400.
   St.dO3[:]   = d_O3(St.ps)  / 86400.
   St.dNOx[:]  = d_NOx(St.ps) / 86400.
   St.gN2O[:]  = gamma_n2o(st_zs)[:]
   St.gNOx[:]  = 0.
   #St.eN2O[:]  = 5e-9
   if n2o_source:
      St.eN2O[:]  = eps_N2O(St.zs)
   else:
      St.eN2O[:]  = 0.
   #St.eNOx[:]  = 0.
   #St.eNOx[:]  = 1.5e-6
   St.eNOx[:]  = init_eps_NOy(St.zs)

   T, O3, N2O, NOx = St.solve()

   T   = pyg.Var((st_pres, ), name = 'T',   values = T)
   O3  = pyg.Var((st_pres, ), name = 'O3',  values = O3)
   N2O = pyg.Var((st_pres, ), name = 'N2O', values = N2O)

   if config == 'noy':
      NOx = pyg.Var((st_pres, ), name = 'NOx', values = NOx * r)
   else:
      NOx = pyg.Var((st_pres, ), name = 'NOx', values = NOx)

   return St, pyg.asdataset([T, O3, N2O, NOx])
# }}}

def plot_nox_timescales(fig = 7):
# {{{
   St0, ds0 = run_nox_basestate()
   St , ds  = run_nox_qbo()

   st_pres = pyg.Pres(St.ps)
   st_zs = pyg.Height(St.zs)

   eNOx   = pyg.Var((st_pres, ), name = 'eNOx',   values = 1.6e-6 + 0*St.ps)
   eNOy   = pyg.Var((st_pres, ), name = 'eNOy',   values = init_eps_NOy(St.zs))
   eAdv   = pyg.Var((st_pres, ), name = 'eAdv',   values = St.w0[:] / 2e3)
   eTrs   = pyg.Var((st_pres, ), name = 'eTrs',   values = St.omega + 0*St.ps)

   nox = init_NOx_0(St.zs)
   r = init_NOx_per_NOy(St.zs)
   noxrz = nox * np.gradient(np.log(r), St.zs)

   fN2O   = pyg.Var((st_pres, ), name = 'fN2O',   values = St.eN2O * np.absolute(ds.N2O[:]))
   fNOx   = pyg.Var((st_pres, ), name = 'fNOx',   values = np.absolute(St.wp) * init_dNOx_0_dz(St.zs))
   fNOy   = pyg.Var((st_pres, ), name = 'fNOy',   values = np.absolute(St.wp) * (init_dNOx_0_dz(St.zs) - noxrz))

   def to_days(t):
      return 1 / (t * 86400.)

   def to_ppbvperday(t):
      return t * 86400e9

   plt.ioff()

   size = (2.8, 3)
   ylims = (110, 4.8)

   axe = pyg.showlines([to_days(v) for v in [eNOy, eAdv, eTrs, eNOx]], labels = [r'$1/\epsilon_{NO_y}$', r'$D / w_0$', r'$1/\omega$', r'Needed'], size = size)
   axe.setp(ylim = ylims, title = '(a) Timescales', xlabel = r'days', xscale = 'log', xlim = (1, 10e5))
   axe.setp_xaxis(major_formatter = plt.LogFormatterMathtext(),
                  major_locator = plt.LogLocator(base = 10., subs = [1.]))

   axf = pyg.showlines([to_ppbvperday(v) for v in [fN2O, fNOx, fNOy]], labels = [r'N$_2$O ox.', 'Adv. of NO$_x$', 'Eff. Adv. of NO$_x$'], size = size)
   axf.setp(ylim = ylims, title = '(b) Sources', xlabel = r'ppbv day$^{-1}$', xscale = 'log', xlim = [3e-6, 8e-2])
   axf.setp_xaxis(major_formatter = plt.LogFormatterMathtext(), 
                  major_locator = plt.LogLocator(base = 10.), minor_locator = plt.LogLocator(base=10.))

   axs = pyg.plot.grid([[axe, axf]])

   plt.ion()
   axs.render(fig)
# }}}

def plot_nox_coefs(fig = 4):
# {{{
   St, ds0 = run_nox_basestate()
   #St , ds  = run_nox_qbo()

   St.dN2Odz[:] = init_dN2O_0_dz(St.zs)
   St.dNOxdz[:] = init_dNOx_0_dz(St.zs)

   st_pres = pyg.Pres(St.ps)
   st_zs = pyg.Height(St.zs)

   nox = init_NOx_0(St.zs)
   r = init_NOx_per_NOy(St.zs)
   noxrz = nox * np.gradient(np.log(r), St.zs)

   N2O_0  = pyg.Var((st_pres, ), name = 'N2O_0', values = init_N2O_0(St.zs))
   NOx_0  = pyg.Var((st_pres, ), name = 'NOx_0', values = nox)

   dN2Odz = pyg.Var((st_pres, ), name = 'dN2Odz', values = St.dN2Odz)
   dNOxdz = pyg.Var((st_pres, ), name = 'dNOxdz', values = St.dNOxdz)
   dNOxdzp = pyg.Var((st_pres, ), name = 'NOxdzp', values = St.dNOxdz - noxrz)
   gN2O   = pyg.Var((st_pres, ), name = 'gN2O',   values = St.gN2O)
   eN2O   = pyg.Var((st_pres, ), name = 'eN2O',   values = St.eN2O)
   eNOy   = pyg.Var((st_pres, ), name = 'eNOy',   values = St.eNOx)

   plt.ioff()

   size = (2.8, 3)
   ylims = (110, 4.8)

   axn0 = pyg.showlines([ds0.N2O, N2O_0], labels = ['calculated', 'fit'], fmts = ['C0', 'k--'], size = size)
   axn0.setp(ylim = ylims, title = r'(a) N$_2$O', xlabel = 'vmr')

   axx0 = pyg.showlines([ds0.NOx, NOx_0], labels = ['calculated', 'fit'], fmts = ['C0', 'k--'], size = size)
   axx0.setp(ylim = ylims, title = r'(b) NO$_x$', xlabel = 'vmr')

   axn = pyg.showlines([ds0.N2O.deriv('pres', dx=st_zs), dN2Odz], labels = ['calculated', 'fit'], fmts = ['C0', 'k--'], size = size)
   axn.setp(ylim = ylims, title = r'(c) $\partial_z$ N$_2$O', xlabel = 'vmr m$^{-1}$', xlim = [-5e-11, 5e-11])

   axx = pyg.showlines([ds0.NOx.deriv('pres', dx=st_zs), dNOxdz, dNOxdzp], labels = ['calculated', 'fit', 'effective'], fmts = ['C0', 'k--', 'C1--'], size = size)
   axx.setp(ylim = ylims, title = r'(d) $\partial_z$ NO$_x$', xlabel = 'vmr m$^{-1}$')
   axx.setp_xaxis(major_formatter = plt.LogFormatter(), major_locator = plt.MultipleLocator(0.5e-12))

   axg = pyg.showlines([gN2O, eNOy], labels = [r'$\gamma_{N_2O}$', r'$\epsilon_{NO_y}$'], size = size)
   axg.setp(ylim = ylims, title = r'(e) $\gamma_{N_2O}$, $\epsilon_{NO_y}$', xlabel = 's$^{-1}$', xscale = 'log', xlim = (1e-12, 1e-6))
   axg.setp_xaxis(major_formatter = plt.LogFormatter(),
                  major_locator = plt.LogLocator())

   axe = pyg.showvar(eN2O, size = size)
   axe.setp(ylim = ylims, title = r'(f) $\epsilon_{N_2O}$', xlabel = 'vmr NO$_x$ (vmr N$_2$O s)$^{-1}$')

   axs = pyg.plot.grid([[axn0, axx0], [axn, axx], [axg, axe]])

   plt.ion()
   axs.render(fig)
# }}}

def plot_nox_qbo(config = 'nox', fig = 3):
# {{{
   run = 'ref'
   dnx = open_dnox_file()
   dn = fit_amp_phase(dnx.dnox)

   dsp = open_pop_file(run)

   St, ds = run_nox_qbo(config)
   St1, ds1 = run_nox_qbo(config, n2o_source = False)

   r = init_NOx_per_NOy(St.zs)
   rz = np.gradient(np.log(r), St.zs)

   nox0 = St.wp * init_dNOx_0_dz(St.zs) / St.eNOx
   nox1 = St.wp * init_NOx_0(St.zs) * rz / St.eNOx

   dsn = to_amp_phase(ds.N2O)
   dsx = to_amp_phase(ds.NOx)
   dsx1 = to_amp_phase(ds1.NOx)

   wp = dsp.resw_pop1 * 1e3
   daW = fit_amp_phase(wp.interpolate('pres', ds.pres))
   psp = daW.phase#(pres=30)[0]

   #drW = fit_amp_phase(wp.interpolate('pres', dn.pres))
   #psr = drW.phase#(pres=30)[0]
   def phs(d, offset): return (d.phase - offset) / np.pi

   plt.ioff()

   def make_pair(ds, var, unit, c, dsref = None, ds1 = None):
      axa = pyg.plot.AxesWrapper(size=(2.5, 3))
      axp = pyg.plot.AxesWrapper(size=(2.5, 3))

      pyg.vplot(1e9*ds.amp(),  c = c, ls = '-',  lw = 2., label = 'calculated', axes = axa)
      pyg.vplot(phs(ds, psp),  c = c, ls = '-', lw = 2., axes = axp)

      if ds1 is not None:
         pyg.vplot(1e9*ds1.amp(),  c = 'C1', ls = '-',  lw = 2., label = 'no N2O', axes = axa)
         pyg.vplot(phs(ds1, psp),  c = 'C1', ls = '-', lw = 2., axes = axp)

      if dsref is not None:
         pyg.vplot(1e9*dsref.amp, c = 'k', ls = '--', lw = 2, label = 'OSIRIS', axes = axa)
         pyg.vplot(phs(dsref.interpolate('pres', psp.pres), psp), c = 'k', ls = '--', lw = 2, axes = axp)

      axa.setp(title = f'{var}: Amplitude', xlabel = unit)
      axa.legend(loc='best', frameon=False)
      axp.setp(title = rf"{var}: Phase (rel to $w'$)", xlabel = '')
      set_rad_axis(axp)

      return [axa, axp]

   axs = []
   axs.append(make_pair(dsn, r'N$_2$O', 'ppbv', 'C0'))
   axs.append(make_pair(dsx, r'NO$_x$', 'ppbv', 'C0', dsref = dn(pres = (110, 4.8)), ds1 = dsx1))

   axs[1][0].setp(xlim = (0, 5))
   
   ax = pyg.plot.grid(axs)
   plt.ion()

   ax.render(fig)
# }}}
