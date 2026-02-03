import numpy as np
import scipy
H = 7000.

def a_T(p):
# {{{
   z = -H * np.log(p/100) * 1e-3
   return 0.04 + 0.002*z + 0.0003*z**2
# }}}

def a_O3(p, a1 = 1/80., a2 = 0.9, a3 = 0.5, a4 = 2.7):
# {{{
   z = -H * np.log(p/100) * 1e-3
   return a1 * z + (a2 + a3*z)*np.exp(-z/a4)
# }}}

def d_T(p, e2 = 0.036):
# {{{
   z = -H * np.log(p/100) * 1e-3
   return -(e2*z)**8
# }}}

def d_O3(p, d1 = 0.01, d2 = 0.046):
# {{{
   z = -H * np.log(p/100) * 1e-3
   return d1 + (d2*z)**6
# }}}

def d_NOx(p, n2 = 0.00007, n3 = 0.0006):
# {{{
   z = -H * np.log(p/100) * 1e-3
   return n2*z + n3*z**2
# }}}

def eps_N2O(z):
# {{{
    # Reaction N2O + O(1D) -> 2 NO
    # Brasseur and Solomon 5.127a (p 329)
    # Rate coefficient from Appendix 3, p 607
    # no temperature dependence
    k_b39 = 6.7e-11 # cm**3 molec**-1 s**-1

    # linearization of rate of production of NO
    # Reaction 5.128
    # P_NO = 2 k_b39 [N2O] [O(1D)]

    number_density_O1D = init_number_density_O1D(z)

    eps_N2O = 2 * k_b39 * number_density_O1D

    return eps_N2O
# }}}

def init_number_density_O1D(z):
# {{{
    # INPUT: z [meters]
    # OUTPUT: number density of O(1D) in molec cm-3

    # from Table A.6.2.c of Brasseur and Solomon (2005)
    z_logp_BS05 = np.arange(0,110001,step=5000) # meters
    O1D_BS05 = np.array([0.,0.,0.,0.,0.,2.9e0,1.3e1,4.0e1,1.0e2,1.8e2,1.9e2,1.5e2,9.8e1,5.2e1,2.4e1,1.3e1,1.5e1,3.3e1,7.0e1,8.9e1,1.4e2,3.2e2,7.8e2])
    # extended downwards with zeros when BS05 has no data to lead to, in effect, no N2O oxidation at 20 km

    interp_O1D_BS05 = scipy.interpolate.interp1d(z_logp_BS05,O1D_BS05,bounds_error=False,fill_value=0.,kind='quadratic')
    number_density_O1D = interp_O1D_BS05(z)
    number_density_O1D = np.clip(number_density_O1D,0,None)

    return number_density_O1D
# }}}

def init_NOx_0(z):
# {{{
    # INPUT: z [meters]
    # OUTPUT: NOx_0 [vmr]

    # digitization of WACCM NOx profile from Park et al., 2017
    # "Variability of Stratospheric Reactive Nitrogen and Ozone Related to the QBO"

    H = 7000. # meters

    p_Park17= np.array([385.2,208.4,132.9,97.83,72.76,49.03,32.32,22.26,15.67,11.15,7.519,4.849,2.993,1.868,1.128]) # hPa
    ps_ref = 1000. # hPa
    z_logp_Park17 = H * np.log(ps_ref/p_Park17)

    # volume mixing ratio [ppbv]
    NOx_ppb = np.array([0.0450,0.0768,0.1192,0.2195,0.4366,0.731,1.091,1.899,3.56,6.57,11.21,15.51,17.3,17.0,13.83]) # ppbv

    interp_NOx = scipy.interpolate.interp1d(z_logp_Park17,NOx_ppb,bounds_error=False,kind='quadratic')

    NOx_0 = interp_NOx(z)

    ppb2vmr = 1e-9 # conversion factor
    NOx_0_vmr = np.clip(NOx_0,0,None)*ppb2vmr # clip up to zero, convert from ppb to vmr

    return NOx_0_vmr
# }}}

def init_dNOx_0_dz(z):
# {{{
    # INPUT: z [meters]
    # OUTPUT: dNOx_0_dz [vmr m**-1]

    z_coarse = np.linspace(17000,50000,21)
    dz_coarse = z_coarse[1]-z_coarse[0]

    NOx_0_vmr = init_NOx_0(z_coarse)

    dNOx_0_vmr_dz = np.gradient(NOx_0_vmr,dz_coarse)

    interp_dNOx_0_vmr_dz= scipy.interpolate.interp1d(z_coarse,dNOx_0_vmr_dz,bounds_error=False, fill_value = (dNOx_0_vmr_dz[0], dNOx_0_vmr_dz[-1]))

    dNOx_0_vmr_dz = interp_dNOx_0_vmr_dz(z)

    return dNOx_0_vmr_dz
# }}}

def init_N2O_0(z):
# {{{
    # INPUT: z [meters]
    # OUTPUT: N2O_0 [vmr]

    # digitization of WACCM NOx profile from Park et al., 2017
    # "Variability of Stratospheric Reactive Nitrogen and Ozone Related to the QBO"

    H = 7000. # meters

    p_Park17 = np.array([381.876,316.91,260.131,213.524,177.200,147.055,120.707,
                100.173,82.225,68.237,56.629,46.48,37.738,31.663,25.707,
                21.3339,17.7046,14.6928,12.0603,10.0086,8.2154,6.81783,
                5.5962,4.5936,3.81215,3.1636,2.5968,2.15505,1.7884,1.4519,
                1.2049,1.01102])

    ps_ref = 1000. # hPa
    z_logp_Park17 = H * np.log(ps_ref/p_Park17)

    # volume mixing ratio [ppbv]
    N2O_ppb =np.array([349.8637,349.8633,349.8633,349.8637,349.8637,347.718,
              349.8637,347.7185,343.467,335.120,326.976,320.998,313.198,
              305.5866,292.709,280.3744,265.276,247.9234,226.073,199.9078,
              172.473,146.083,119.9845,95.5641,76.114,59.8813,46.2491,
              35.72036,26.58986,18.726,12.71004,8.62741])

    interp_N2O = scipy.interpolate.interp1d(z_logp_Park17,N2O_ppb,bounds_error=False,kind='quadratic')

    N2O_0 = interp_N2O(z)

    ppb2vmr = 1e-9 # conversion factor
    N2O_0_vmr = np.clip(N2O_0,0,None)*ppb2vmr # clip up to zero, convert from ppb to vmr

    return N2O_0_vmr
# }}}

def init_dN2O_0_dz(z):
# {{{
    # INPUT: z [meters]
    # OUTPUT: dN2O_0_dz [vmr m**-1]

    z_coarse = np.linspace(17000,50000,21)
    dz_coarse = z_coarse[1]-z_coarse[0]

    N2O_0_vmr = init_N2O_0(z_coarse)

    dN2O_0_vmr_dz = np.gradient(N2O_0_vmr,dz_coarse)

    interp_dN2O_0_vmr_dz= scipy.interpolate.interp1d(z_coarse,dN2O_0_vmr_dz,bounds_error=False, fill_value = (dN2O_0_vmr_dz[0], dN2O_0_vmr_dz[-1]))

    dN2O_0_vmr_dz = interp_dN2O_0_vmr_dz(z)

    return dN2O_0_vmr_dz
# }}}

def init_eps_NOy(z):
# {{{
    # INPUT: z [meters]
    # OUTPUT: eps_NOy for linear model [s**-1]

    eps_NOy_WACCM = np.array([2.93153939e-04, 3.40709666e-04, 2.66387025e-04, 1.78474394e-04,
                           1.07188549e-04, 6.30159662e-05, 3.94026459e-05, 2.61252665e-05,
                           1.65738584e-05, 9.70159351e-06, 6.42008961e-06, 5.21610205e-06,
                           4.72063707e-06, 4.17067936e-06, 3.13769453e-06, 2.02433616e-06,
                           1.04422203e-06, 6.55871317e-07, 4.96386455e-07, 4.04176090e-07,
                           3.17944330e-07, 2.53793772e-07, 2.00721346e-07, 1.52599960e-07,
                           1.13493847e-07, 8.72400791e-08, 7.13009697e-08, 6.13507530e-08,
                           5.60982763e-08, 5.13082065e-08, 4.33159074e-08, 3.23497787e-08,
                           2.13097995e-08, 1.26018330e-08, 6.42271906e-09, 2.78322765e-09,
                           1.00313161e-09, 3.15377589e-10, 9.49195060e-11, 3.55718084e-11,
                           2.25634092e-11, 2.27992321e-11, 3.09165360e-11, 4.79215588e-11,
                           8.22172716e-11, 1.60732437e-10, 3.29729628e-10, 5.99945059e-10,
                           6.49399627e-10, 4.36311826e-10, 2.12506616e-10, 8.62017488e-11,
                           3.15115124e-11, 1.06836814e-11, 3.43168056e-12, 1.15153137e-12,
                           4.01103876e-13, 1.62638953e-13, 7.59899040e-14, 4.00745107e-14,
                           2.31835732e-14, 1.29506786e-14, 9.13675186e-15, 8.30049965e-15,
                           8.10747762e-15, 7.52770358e-15, 6.54913832e-15, 5.67422082e-15,
                           5.27490796e-15, 2.12466679e-15])


    plev_WACCM = np.array([5.960300e-06, 9.826900e-06, 1.620185e-05, 2.671225e-05, 4.404100e-05,
       7.261275e-05, 1.197190e-04, 1.973800e-04, 3.254225e-04, 5.365325e-04,
       8.846025e-04, 1.458457e-03, 2.404575e-03, 3.978250e-03, 6.556826e-03,
       1.081383e-02, 1.789800e-02, 2.955775e-02, 4.873075e-02, 7.991075e-02,
       1.282732e-01, 1.981200e-01, 2.920250e-01, 4.101675e-01, 5.534700e-01,
       7.304800e-01, 9.559475e-01, 1.244795e+00, 1.612850e+00, 2.079325e+00,
       2.667425e+00, 3.404875e+00, 4.324575e+00, 5.465400e+00, 6.872850e+00,
       8.599725e+00, 1.070705e+01, 1.326475e+01, 1.635175e+01, 2.005675e+01,
       2.447900e+01, 2.972800e+01, 3.592325e+01, 4.319375e+01, 5.167750e+01,
       6.152050e+01, 7.375096e+01, 8.782123e+01, 1.033171e+02, 1.215472e+02,
       1.429940e+02, 1.682251e+02, 1.979081e+02, 2.328286e+02, 2.739108e+02,
       3.222419e+02, 3.791009e+02, 4.459926e+02, 5.246872e+02, 6.097787e+02,
       6.913894e+02, 7.634045e+02, 8.208584e+02, 8.595348e+02, 8.870202e+02,
       9.126445e+02, 9.361984e+02, 9.574855e+02, 9.763254e+02, 9.925561e+02]) # hPa


    H = 7000. # meters
    ps_ref = 1000. # hPa
    z_logp = H * np.log(ps_ref/plev_WACCM)

    interp_eps_NOy = scipy.interpolate.interp1d(z_logp,eps_NOy_WACCM,bounds_error=False,kind='quadratic')

    eps_NOy_0 = interp_eps_NOy(z)

    return eps_NOy_0
# }}}

def init_NOx_per_NOy(z):
# {{{
    # INPUT: z [meters]
    # OUTPUT: NOx/NOy at z [nondimensionless]

    NOx_per_NOy_MIROC = np.array([0.41329309, 0.31962148, 0.18948178, 0.15351273, 0.20495902,
       0.27530478, 0.38177958, 0.38815039, 0.41348385, 0.45093616,
       0.48484818, 0.51401524, 0.54138559, 0.5497206 , 0.53472018,
       0.49515413, 0.42988934, 0.34338881, 0.26090937, 0.20561341,
       0.17273046, 0.18690681, 0.20960833, 0.23814653, 0.29622721,
       0.35638977, 0.41101973, 0.47847075, 0.55705619, 0.5998289 ,
       0.64422139, 0.68950104, 0.77879978, 0.82117371, 0.85928701,
       0.89570808, 0.93313446, 0.9539997 , 0.97543168, 0.98671179,
       0.8678775 , 0.        ]) # dimensionless

    plev_MIROC = np.array([1.00e+05, 9.25e+04, 8.50e+04, 7.00e+04, 6.00e+04, 5.00e+04, 4.00e+04,
       3.00e+04, 2.50e+04, 2.00e+04, 1.70e+04, 1.50e+04, 1.30e+04, 1.15e+04,
       1.00e+04, 9.00e+03, 8.00e+03, 7.00e+03, 6.00e+03, 5.00e+03, 4.00e+03,
       3.50e+03, 3.00e+03, 2.50e+03, 2.00e+03, 1.70e+03, 1.50e+03, 1.30e+03,
       1.10e+03, 1.00e+03, 9.00e+02, 8.00e+02, 6.00e+02, 5.00e+02, 4.00e+02,
       3.00e+02, 2.00e+02, 1.50e+02, 1.00e+02, 7.00e+01, 5.00e+01, 4.00e+01])
    plev_MIROC /= 1e2 # convert to hPa

    H = 7000. # meters
    ps_ref = 1000. # hPa
    z_logp = H * np.log(ps_ref/plev_MIROC)

    interp_NOx_per_NOy = scipy.interpolate.interp1d(z_logp,NOx_per_NOy_MIROC,bounds_error=False,kind='quadratic')

    NOx_per_NOy_0 = interp_NOx_per_NOy(z)

    return NOx_per_NOy_0
# }}}

def init_HNO3_per_NOy(z):
# {{{
    # INPUT: z [meters]
    # OUTPUT: HNO3/NOy at z [nondimensionless]

    HNO3_per_NOy_MIROC = np.array([5.79805779e-01, 6.74489679e-01, 8.02461891e-01, 8.34585931e-01,
       7.73685215e-01, 6.82986534e-01, 5.15379608e-01, 3.22149519e-01,
       2.99179642e-01, 3.31653870e-01, 3.60712930e-01, 3.77863874e-01,
       3.91612085e-01, 4.04741617e-01, 4.32133931e-01, 4.74270078e-01,
       5.36149707e-01, 6.12603709e-01, 6.81775834e-01, 7.26568181e-01,
       7.50740652e-01, 7.10481197e-01, 6.59116086e-01, 6.05098788e-01,
       5.18969643e-01, 4.46263108e-01, 3.88576621e-01, 3.25110920e-01,
       2.58174919e-01, 2.24068479e-01, 1.89937348e-01, 1.56148723e-01,
       9.18381915e-02, 6.23096934e-02, 3.86465517e-02, 1.91788268e-02,
       6.07755094e-03, 2.45411843e-03, 7.08242052e-04, 3.21586901e-04,
       1.65942085e-04, 1.08636733e-04]) # dimensionless

    plev_MIROC = np.array([1.00e+05, 9.25e+04, 8.50e+04, 7.00e+04, 6.00e+04, 5.00e+04, 4.00e+04,
       3.00e+04, 2.50e+04, 2.00e+04, 1.70e+04, 1.50e+04, 1.30e+04, 1.15e+04,
       1.00e+04, 9.00e+03, 8.00e+03, 7.00e+03, 6.00e+03, 5.00e+03, 4.00e+03,
       3.50e+03, 3.00e+03, 2.50e+03, 2.00e+03, 1.70e+03, 1.50e+03, 1.30e+03,
       1.10e+03, 1.00e+03, 9.00e+02, 8.00e+02, 6.00e+02, 5.00e+02, 4.00e+02,
       3.00e+02, 2.00e+02, 1.50e+02, 1.00e+02, 7.00e+01, 5.00e+01, 4.00e+01])
    plev_MIROC /= 1e2 # convert to hPa

    H = 7000. # meters
    ps_ref = 1000. # hPa
    z_logp = H * np.log(ps_ref/plev_MIROC)

    interp_HNO3_per_NOy = scipy.interpolate.interp1d(z_logp,HNO3_per_NOy_MIROC,bounds_error=False,kind='quadratic')

    HNO3_per_NOy_0 = interp_HNO3_per_NOy(z)

    return HNO3_per_NOy_0
# }}}
