#######################################################################
#
# SiRaChA: Simultaneous Radiative and 
# Chemical relaxation of the Atmosphere
#
# Author: Edward Charlesworth
#
# Started: January 2018
#
#######################################################################

#######################################################################
#
# The idea behind the structure of this code is that the model is a
# class which contains all the methods (functions) that it needs. You
# /can/ initialize the column object with a dictionary of input data,
# but you would have to know exactly what the structure should be.
# What you should do instead is initialize the model with a string
# which is a filepath to an initialization file. See the example
# script in the work directory for more information.
#
#######################################################################

#Dependencies:
import numpy as np
from netCDF4 import Dataset
import os
import time
from collections.abc import Iterable
import sys
import pygeode as pyg

class column :

    def __init__( self , input_data , inits_extra=None , verbose=False ):

        # Logical options
        self.bool_timing_output=True        # Output model timing info
        self.bool_ozone_integration=True    # Integrate ozone
        self.bool_temps_convection=True     # You probably want this
        self.bool_run_rad=True              # Model runs faster if it's
                                            # only in ozone mode
        self.bool_fdh=False                 # Run as FDH. Turn chem off and transport of ozone off.
        self.bool_impose_resw=False         # Impose dynamical heating
        self.bool_pce_only=False            # Run PCE only. Read rad heating from input file and do not change.
                                            # (with transport of ozone)
        self.resw_in = pyg.open('resw.nc').resw #half resw variability
        self.nox_in = pyg.open('nox.nc').dnox
        
        #
        # Don't mess with the rest of the stuff in this method.
        #

        self.timing_output_length = 0

        # Set up the (I)nitial condition and current timepstep (D)ata dicts.
        if type( input_data ) is dict:
            
            self.extra = inits_extra.copy()
            self.I = inits.copy()
            self.D = inits.copy()
            self.O = None

        elif type( input_data ) is str:

            self.I = dict()
            self.D = dict()
            self.O = None
            self.extra = dict()
            
            nc_fid = Dataset( input_data , 'r' )
            nc_fid.set_always_mask(False)
            keys = list(nc_fid.variables.keys())
            for key in keys:
                self.I[key] = nc_fid.variables[key][:].squeeze()
                self.D[key] = nc_fid.variables[key][:].squeeze()
                extra = []
                if self.check_attr(nc_fid.variables[key].form):
                  form = self.check_attr(nc_fid.variables[key].form)
                else:
                  form = 'constant'

                extra.append(form)
                #extra.append(self.check_attr(nc_fid.variables[key].form))
                extra.append(self.check_attr(nc_fid.variables[key].units))
                extra.append(self.check_attr(nc_fid.variables[key].recording))
                self.extra[key] = extra
                if verbose: print( key + "  " + str(extra) ) 
            nc_fid.close()

            #self.extra = inits_extra.copy()

        else:

            raise Exception("Inits must be either a str or a dict!")

        # Set up experimental variables list
        self.exp_strs = {}

    def check_attr( self , attr ):
        if attr == "None": 
            return None
        elif attr == "False":
            return False
        elif attr == "True":
            return True
        else:
            return attr

    def change_value( self , variable , value ):

        if hasattr( value , 'copy' ):
            self.I[variable] = value.copy()
            self.D[variable] = value.copy()
        else:
            self.I[variable] = value
            self.D[variable] = value

    def purge_variable( self , variable ):

        self.I.pop(variable)
        self.D.pop(variable)
        self.extra.pop(variable)

    def add_variable( self , variable , value , extra_values ):

        # form units recording
        # Extra: form - {None,'lv','ly'}, units, recording - {True,False}
        if hasattr( value , 'copy' ):
            self.I[variable] = value.copy()
            self.D[variable] = value.copy()
        else:
            self.I[variable] = value
            self.D[variable] = value
        self.extra[variable] = extra_values

    def timer( self , total , done ):

        gone = self.runtime
        # Print time remaining info to user
        left = gone/(done+1)*(total-done)
        percent = (done+1)/total*100
        if left < 60: 
            out = left
            typ = "seconds"
        elif left < 3600: 
            out = left/60
            typ = "minutes"
        else:
            out = left/60/60
            typ = "hours"

        print( " "*self.timing_output_length , end="\r" )
        if self.timing_output_length == 0: print("")

        message = "Percent done, time left: %6.2f%% %5.2f %s"\
                % ( percent , out , typ )

        self.timing_output_length = len(message)

        print(message,end="\r")

    def add_ly_dim( self, data ):

        data = np.expand_dims( data , len(data.shape) )
        data = np.repeat( data , self.D['nLayers'] , axis = len(data.shape)-1 )
        return data

    def ozone_chemistry_initialize( self ):

        # Cross-sections for ozone j3 coefficient
        self.sigma_j3_o3 = np.array([ \
                 31.5,32.6,36.3,43.3,53.9,69.3,90.3,118,154,199,255,322,401,490
                ,590,693,802,908,1001,1080,1125,1148,1122,1064,968,840,698,547
                ,406,282,184,113,65.1,45.2,39.2,34.3,30.3,26.2,23.4,20.1,17.9
                ,15.5,13.5,12.2,10.2,9.24,7.95,6.91,6.25,5.19,4.77,4.02,3.72
                ,2.89,2.99,2.1,2.05,1.41,1.01,0.697,0.32,0.146,0.0779,0.0306
                ,0.0136,0.00694,0.00305,0.0013,0.00085,0.000572,0.000542
                ,0.000668,0.000956,0.00115,0.00158,0.00258,0.00295,0.00393
                ,0.00656,0.00697,0.00882,0.0137,0.0165,0.0185,0.0218,0.0366
                ,0.0367,0.041,0.0481,0.0754,0.0813,0.0816,0.0908,0.121,0.16
                ,0.158,0.166,0.183,0.219,0.267,0.287,0.295,0.319,0.337,0.358
                ,0.398,0.439,0.467,0.481,0.464,0.446,0.447,0.476,0.513,0.514
                ,0.478,0.438,0.406,0.382,0.356,0.327,0.297,0.271,0.251,0.231
                ,0.21,0.19,0.17,0.151,0.137,0.126,0.113,0.0989,0.0868,0.0784
                ,0.0731,0.0696,0.0622,0.0543,0.0478,0.0442,0.0432,0.0447
                ,0.0425,0.0338,0.0286,0.0262,0.026,0.0294,0.0318,0.0262,0.0208
                ,0.0173,0.0157,0.0156,0.0186,0.0221,0.0206,0.0145
                ])*1E-20*1E-4 #m^2

        self.flux_j3=np.array([
        1.52000000e+16,1.78000000e+16,2.20000000e+16,
        2.69000000e+16,4.54000000e+16,7.14000000e+16,
        8.35000000e+16,8.39000000e+16,1.08000000e+17,
        1.18000000e+17,1.60000000e+17,1.34000000e+17,
        1.41000000e+17,1.57000000e+17,1.38000000e+17,
        1.60000000e+17,1.45000000e+17,2.20000000e+17,
        1.99000000e+17,1.97000000e+17,1.94000000e+17,
        2.91000000e+17,4.95000000e+17,1.07000000e+18,
        1.20000000e+18,1.10000000e+18,1.04000000e+18,
        1.52000000e+18,8.24000000e+17,1.52000000e+18,
        2.15000000e+18,3.48000000e+18,3.40000000e+18,
        3.29129713e+18,3.25104354e+18,3.27091236e+18,
        3.49083615e+18,3.71075993e+18,3.93068372e+18,
        4.15060751e+18,4.32716579e+18,4.47922492e+18,
        4.63128405e+18,4.78334319e+18,4.93540232e+18,
        5.04032626e+18,5.14024470e+18,5.24016313e+18,
        5.34008157e+18,5.44000000e+18,5.53800000e+18,
        5.63600000e+18,5.73400000e+18,5.83200000e+18,
        5.93000000e+18,6.13400000e+18,6.44000000e+18,
        6.84800000e+18,7.31000000e+18,8.15000000e+18,
        7.81000000e+18,8.35000000e+18,8.14000000e+18,
        8.53000000e+18,9.17000000e+18,8.38000000e+18,
        1.04000000e+19,1.10000000e+19,9.79000000e+18,
        1.13000000e+19,8.89000000e+18,1.14000000e+19,
        9.17000000e+18,1.69000000e+19,1.70000000e+19,
        1.84000000e+19,1.87000000e+19,1.95000000e+19,
        1.81000000e+19,1.67000000e+19,1.98000000e+19,
        2.02000000e+19,2.18000000e+19,2.36000000e+19,
        2.31000000e+19,2.39000000e+19,2.38000000e+19,
        2.39000000e+19,2.44000000e+19,2.51000000e+19,
        2.30000000e+19,2.39000000e+19,2.48000000e+19,
        2.40000000e+19,2.46000000e+19,2.49000000e+19,
        2.32000000e+19,2.39000000e+19,2.42000000e+19,
        2.55000000e+19,2.51000000e+19,2.49000000e+19,
        2.55000000e+19,2.53000000e+19,2.54000000e+19,
        2.50000000e+19,2.57000000e+19,2.58000000e+19,
        2.67000000e+19,2.67000000e+19,2.70000000e+19,
        2.62000000e+19,2.69000000e+19,2.63000000e+19,
        2.68000000e+19,2.66000000e+19,2.59000000e+19,
        2.69000000e+19,2.61000000e+19,2.62000000e+19,
        2.62000000e+19,2.63000000e+19,2.60000000e+19,
        2.55000000e+19,2.48000000e+19,2.57000000e+19,
        2.61000000e+19,2.61000000e+19,2.62000000e+19,
        2.62000000e+19,2.57000000e+19,2.52000000e+19,
        2.60000000e+19,2.58000000e+19,2.52000000e+19,
        2.51000000e+19,2.48000000e+19,2.45000000e+19,
        2.48000000e+19,2.45000000e+19,2.44000000e+19,
        2.39000000e+19,2.40000000e+19,2.41000000e+19,
        2.40000000e+19,2.38000000e+19,2.34000000e+19,
        2.32000000e+19,2.30000000e+19,2.33000000e+19,
        2.34000000e+19,2.29000000e+19,2.29000000e+19,
        2.27000000e+19,2.27000000e+19,2.20000000e+19,
        2.22000000e+19,2.18000000e+19,2.20000000e+19])
        self.flux_j3 = self.add_ly_dim( self.flux_j3 )
        self.sigma_j3_o3 = self.add_ly_dim( self.sigma_j3_o3 )
        
        # Herzberg
        self.sigma_hz_o3=np.array([  
            3.75053634e-23,   4.07987062e-23,   4.44747926e-23,
            4.93595853e-23,   5.43898263e-23,   6.13377171e-23,
            6.82856079e-23,   7.72213781e-23,   8.64969965e-23,
            9.73672432e-23,   1.09345622e-22,   1.22227677e-22,
            1.37462548e-22,   1.52697419e-22,   1.71036845e-22,
            1.89666529e-22,   2.10315651e-22,   2.32992306e-22,
            2.55782617e-22,   2.82312017e-22,   3.08841418e-22,
            3.37414634e-22,   3.67999226e-22,   3.98583817e-22,
            4.32019489e-22,   4.65699716e-22,   5.00297652e-22,
            5.37273063e-22,   5.74248475e-22,   6.11355247e-22,
            6.48559509e-22,   6.85763771e-22,   7.23969841e-22,
            7.62417813e-22,   8.00865785e-22])# m^2
        self.hz_flux=np.array([  
            2.28437544e+16,   2.51490943e+16,   2.89503456e+16,
            3.74756912e+16,   4.62269795e+16,   5.79571848e+16,
            6.96873900e+16,   7.59642226e+16,   8.13087456e+16,
            8.36020541e+16,   8.37750270e+16,   8.67301947e+16,
            9.69290944e+16,   1.07127994e+17,   1.11785966e+17,
            1.15925895e+17,   1.26486738e+17,   1.43494230e+17,
            1.59696298e+17,   1.49401307e+17,   1.39106316e+17,
            1.35365854e+17,   1.38075881e+17,   1.40785908e+17,
            1.46576537e+17,   1.52631410e+17,   1.55043446e+17,
            1.48018118e+17,   1.40992790e+17,   1.42561315e+17,
            1.50507856e+17,   1.58454398e+17,   1.55738095e+17,
            1.50447090e+17,   1.45156085e+17])# photons / m^2 / s
        self.sigma_hz_o3 = self.add_ly_dim( self.sigma_hz_o3 )
        self.hz_flux = self.add_ly_dim( self.hz_flux )

        # Schumann-Runge Bands Calculation
        self.sigma_srb_o2 = self.add_ly_dim(np.array([
            [1.03E-21,1.75E-21,4.59E-21,1.71E-20,1.01E-19,1.10E-18],
            [7.68E-22,1.24E-21,2.67E-21,1.08E-20,8.36E-20,7.70E-19],
            [1.13E-21,2.02E-21,4.66E-21,1.65E-20,9.30E-20,5.02E-19],
            [5.56E-22,1.58E-21,3.72E-21,1.38E-20,7.22E-20,3.44E-19],
            [2.97E-22,5.83E-22,2.05E-21,8.19E-21,4.80E-20,2.66E-19],
            [1.35E-22,2.99E-22,7.33E-22,3.07E-21,1.69E-20,1.66E-19],
            [1.08E-22,2.09E-22,5.88E-22,2.59E-21,1.58E-20,1.03E-19],
            [4.49E-23,6.68E-23,2.38E-22,1.13E-21,6.99E-21,5.55E-20],
            [1.91E-23,3.44E-23,1.17E-22,4.74E-22,2.65E-21,2.53E-20],
            [1.12E-23,2.45E-23,7.19E-23,3.04E-22,1.75E-21,1.11E-20],
            [9.60E-24,1.26E-23,2.55E-23,1.05E-22,5.19E-22,2.82E-21],
            [6.82E-24,7.50E-24,1.03E-23,2.48E-23,1.52E-22,1.25E-21],
            [6.98E-24,7.49E-24,9.31E-24,1.65E-23,7.87E-23,4.63E-22],
            [6.74E-24,6.77E-24,6.93E-24,7.36E-24,1.04E-23,5.18E-23],
            [6.84E-24,6.85E-24,6.88E-24,7.11E-24,8.41E-24,2.87E-23]
            ])) /100**2 #m^2
        self.weights_srb = np.array([ 0.05 , 0.20 , 0.25 , 0.25 , 0.20 , 0.05 ])
        self.weights_srb = np.repeat( np.expand_dims( 
                 self.add_ly_dim( self.weights_srb )
                , axis = 0 ) , 15, 0 )
        self.solar_flux_srb = self.add_ly_dim( np.array([  
             1.34125000e+15,1.42562500e+15,1.59024691e+15
            ,1.86000000e+15,1.90090909e+15,1.77508876e+15
            ,1.96017544e+15,2.49457143e+15,3.02606742e+15
            ,3.60895028e+15,3.95142857e+15,5.40437500e+15
            ,6.12000000e+15,6.54320000e+15,7.65058824e+15]) ) # photons m^-2 s^-1 nm^-1
        self.interval_srb = self.add_ly_dim(np.array([
            0.8,1.,1.1,1.2,1.5,1.5,1.7,1.9,2.,2.3,2.2,2.5,1.3,1.5,2.5
            ])) #nm
        self.sigma_srb_o3=self.add_ly_dim( np.array([  
             7.98025000e-23,7.90712500e-23,7.79327160e-23
            ,7.63000000e-23,7.35181818e-23,6.99887574e-23
            ,6.56842105e-23,6.04342857e-23,5.43168539e-23
            ,4.81127072e-23,4.32857143e-23,3.93062500e-23
            ,3.53081633e-23,3.29520000e-23,3.26000000e-23]) )#m^2

        # Other initializations
        self.col_o3_toa   = 0.0# 2.48E+20 # molec / m^2 (Bausseur and Solomon)

    def ozone_chemistry( self ):

        # I'm trying to make all the units here standard scientific units, so
        # that means kg, m, K, etc. Wavelengths are in nanometers.

        # molecule number density
        self.nm = ( 6.022E+23 * self.D['lyP'] )                  \
                / ( self.D['molar_mass_air'] * self.lyT * 287.058 ) 
                #/ ( self.D['molar_mass_air'] * self.D['lyT'] * 287.058 ) 
                # molec/mol * Pa / ( kg/mol * K * J/kg*K )
                # molec/m^3

        # diatomic oxygen cross-section
        sigma0 = np.array([                             \
                  7.71,7.48,7.39,7.19,7.00,6.82,6.54    \
                 ,6.35,6.18,5.99,5.86,5.62,5.39,5.13    \
                 ,4.89,4.70,4.50,4.32,4.11,3.89,3.66    \
                 ,3.42,3.18,2.97,2.82,2.62,2.44,2.28    \
                 ,2.12,1.95,1.80,1.65,1.51,1.38,1.26    \
                 ,1.16])*1E-28 # m^2
        sigma0 = ( sigma0[:-1]+sigma0[1:] ) / 2
        sigma0 = np.expand_dims( sigma0 , 1 )
        sigma0 = np.repeat( sigma0 , len(self.D['lyP']) , axis = 1 )
        dsigmadTorr = np.array([                        \
                  13.7,13.4,12.9,12.5,12.1,11.7,11.5    \
                 ,11.1,10.7,10.3,9.76,9.47,9.11,8.89    \
                 ,8.53,8.10,7.70,7.29,6.96,6.61,6.29    \
                 ,6.01,5.82,5.51,5.16,4.89,4.63,4.38    \
                 ,4.13,3.91,3.68,3.50,3.30,3.07,2.90    \
                 ,2.69])*1E-31 # m^2/torr
        dsigmadTorr = ( dsigmadTorr[:-1]+dsigmadTorr[1:] ) / 2
        dsigmadTorr = np.expand_dims( dsigmadTorr , 1 )
        dsigmadTorr = np.repeat( dsigmadTorr , len(self.D['lyP'] ) , axis = 1 )
        sigma_hz_o2 = ( sigma0 + self.D['lyP']/100/1.33322*dsigmadTorr ) # m^-2
        # cosine phi factor (always shows up as 1/cos(sza))
        cosfac = 1 / np.cos( self.D['sza'] *2*np.pi/360 )
        # densities
        o2 = self.nm*self.D['lyO2']
        # column densities ( molecules / meter^2 )
        nm_col = np.diff(self.D['lvZ']*1000) * self.nm
        o2_col = nm_col * self.D['lyO2']
        #o3_col = nm_col * self.D['lyO']
        o3_col = nm_col * self.lyO
        # over-model-top oxygen and ozone columns
        #col_o2_toa = (self.D['lvP'][-1]*6.022E+23) \
        col_o2_toa = (self.lvP[-1]*6.022E+23) \
                / (9.81*self.D['lyO2'][-1]*0.032) # molec/m^2
        # total overhead column densities ( molecules / meter^2 )
        o2_tcol = cosfac * ( np.flipud( np.cumsum( np.flipud( o2_col ) ) ) + \
                col_o2_toa )
        o3_tcol = cosfac * ( np.flipud( np.cumsum( np.flipud( o3_col ) ) ) + \
                self.col_o3_toa )

        ############################### Radiation ##############################
        # J3 calculation
        o3_tr = np.exp( - o3_tcol * self.sigma_j3_o3 )      # You can consolidate this
        self.j3 = np.sum( o3_tr * self.flux_j3 * self.sigma_j3_o3 , axis=0 ) 

        # Herzberg calculation
        O2Tr = np.exp( - o2_tcol *      sigma_hz_o2 ) \
             * np.exp( - o3_tcol * self.sigma_hz_o3 )
        j2_hz = np.sum( O2Tr * sigma_hz_o2 * self.hz_flux  , axis = 0 )
        self.j2_hz = j2_hz
        # Schumann-Runge bands calculation
        self.o3_tcol = o3_tcol
        tr_srb_o3 = np.exp( - o3_tcol * self.sigma_srb_o3 )
        
        tr_srb_o2 = self.weights_srb \
                * np.exp( - self.sigma_srb_o2 * o2_tcol ) # unitless
        flux_srb = tr_srb_o3 * self.solar_flux_srb * np.sum( tr_srb_o2 , axis = 1 ) 
                # photons / m^2 s nm

        self.j2_srb = np.sum(                                    \
                tr_srb_o3 * self.interval_srb * flux_srb *      \
                np.sum( self.sigma_srb_o2 * tr_srb_o2 , axis = 1 ) , \
                axis = 0 ) # photons / s (quamtum efficiency is 1)

        self.j2 = self.j2_hz + self.j2_srb

        ######################### Oxygen Net Change ###########################
        #
        # From B&S page 273-277 (283-287)
        #
        #   J2: O2 + hv ->  2O
        #   k1: 2O + M  ->  O2 + M              (only relevant in thermosphere)
        #   k3: O3 + O  -> 2O2
        #  J3e: O3 + hv -> O2(3 sigma g) + O(3P)
        #  J3g: O3 + hv -> O2(1 delta g) + O(1D)
        #  J2e: O2 + hv -> O(1D) + O(3P)      (upper part of middle atmosphere)
        #   k5: O(1D) + O3 -> 2O2
        #
        # d(Ox)/dt = 2(J2+J2e)[O2]-2k3[O3][O]-2k5[O(1D)][O3]+2k1[M][O]**2
        # 

        ######################## Oxygen Partitioning ##########################
        #
        # From B&S pages 278-279 (288-289)
        #
        # Odd-oxygen partitioning occurs through the reactions
        #
        #   O3 + hv     -> O2 + O(1D)
        #   O2 + O(1D)  -> O  + O2(1 sigma g)
        #   O  + O2 + M -> O3 + M
        #
        # And this leads to the partitioning ratio
        #
        #   O(1D)/O3 ~ J3*/(k4a*[N2]+k4b*[O2])
        #
        # A ratio for O(3P) to O3 is also given.
        #
        #   O(3P)/O3 ~ JO2/(k2*[M2]*[O2])
        #

        ####################### Nitrogen Partitioning #########################

        #
        # The chemistry:
        #
        #   b3:  NO2 + O  ->  NO  + O2   
        #   b4:  NO  + O3 ->  NO2 + O2
        # 
        #   Assuming steady state, then NO/NO2 = (b3*O)/(b4*O3).
        #   Because we relate O to O3 with o_to_o3_ratio, the O3 concentration
        #   cancels and we're left with b3*o_to_o3_ratio/b4
        #
        # Rates from JPL document 15-10, pages 1-70 and 1-71:
        #
        #   b3 = 5.1E-12*exp[210*T**-1]     cm^3/molec/s (T in K)
        #   b4 = 3.0E-12*exp[-1500*T**-1]   (the same as above)               
        #
        # Chemistry which is not included but which I might include later:
        #
        # JNO2:  NO2 + hv(lambda<405nm) -> NO + O
        #
        # Excluding this means that NO levels are lower during daytime, but at
        # high altitudes they're already dominant (NOX ~= NO) during daytime,
        # so this is really just an effect at lower altitudes where O 
        # concentrations are low.
        #
        # If this was included, steady state gives NO/NO2=(b3*O+JNO2)/(b4*O3).

        #lyNOx = self.nox_interper( self.D['lyP']/100 )
        #lyNOx[self.D['lyP']/100>self.nox_limits[0]]=0
        #lyNOx[self.D['lyP']/100<self.nox_limits[1]]=0
        #self.D['lyNOx'] = lyNOx

        #k2  = (6E-34   * ( self.D['lyT'] / 300 )**(-2.4))/100**6 # (m^3/molec)^2/s
        k2  = (6E-34   * ( self.lyT / 300 )**(-2.4))/100**6 # (m^3/molec)^2/s
        o_to_o3_ratio = self.j3 / (k2*o2*self.nm) # O/O3

        #b3 = 5.1E-12*np.exp(   210 / self.D['lyT'] )/100**3         # (m^3/molec)/s
        #b4 = 3.0E-12*np.exp( -1500 / self.D['lyT'] )/100**3         # (m^3/molec)/s
        b3 = 5.1E-12*np.exp(   210 / self.lyT )/100**3         # (m^3/molec)/s
        b4 = 3.0E-12*np.exp( -1500 / self.lyT )/100**3         # (m^3/molec)/s

        #no_to_no2_ratio = b3*o_to_o3_ratio/b4           # unitless
        no_to_no2_ratio = (b3*o_to_o3_ratio + self.D['jno2']/(self.lyO*self.nm))/b4           # unitless

        if len(np.where(o_to_o3_ratio==0)[0])>0:
            no_to_nox_ratio = 0 
        else:
          no_to_nox_ratio = ( 1 + 1/no_to_no2_ratio)**-1  # unitless

        lyNO  = no_to_nox_ratio * (self.D['lyNOx'] + self.dnox)
        lyNO2 = (1-no_to_nox_ratio) * (self.D['lyNOx'] + self.dnox)
        self.lyNO2 = lyNO2
        
        #
        # The loss of Ox due to NOx is:
        #
        # sink_nox = b4*lyNO*lyO + b3*lyNO2*o_to_o3_ratio*lyO
        #
        # where lyO is the ozone mixing ratio. We'll avoid the application of
        # lyO until the lyO evolution line below, in case we want to apply an
        # internal ozone chemistry time integration routine later on. But to
        # adjust the units we'll also have to multiply by the number density of
        # air, which is self.D['nm'].
        #

        sink_nox = (b4*lyNO + b3*lyNO2*o_to_o3_ratio)*self.nm*86400 # ppv/day

        ####################### Hydrogen Partitioning #########################
        #
        # This component is not implemented, but these comments are here to
        # provide some information about how this could be implemented.
        #
        # First, from Brausseur and Solomon page 323 (PDF 333), we have that
        # the ratio HO2/OH is...
        #
        # HO2/OH = (a5*a1*[M][O2])/(a7*(a1*[M][O2]+a2*[O3]))
        #
        # and that
        #
        # [H] = (a5*[O][OH])/(a1*[M][O2]+a2*[O3])
        #
        # Both valid above 40 km. These three chemicals form the odd-hydrogen
        # family ([HOx]=[H]+[OH]+[HO2]).
        #
        # To compute these, one would need constants for the relevant reactions
        # which are...
        #
        # a1: H   + O2 + M -> HO2 + M 
        # a2: H   + O3     -> O2  + OH 
        # a5: OH  + O      -> O2  + H 
        # a7: HO2 + O      -> O2  + OH
        # 
        # One would also need concentrations for M, O2, HOx (all easy), O, and
        # O3 (latter included, former  already computed for the NOx 
        # calculation). Given that information, one could compute the rate of
        # Ox loss by
        #
        # loss_HOx = a2*[H][O3]+a5*[OH][O] + a7*[HO2][O]
        #    = a2*[
        #
        # For the three simple rate constants, see JPL 15-10 pages 1-54 (68).
        #
        # a2 = 1.4E-10*exp( -470/T )    [cm^3 molecule^-1 s^-1]
        # a5 = 1.8E-11*exp( 180/T )     [ditto]
        # a7 = 3.0E-11*exp( 200/T )     [ditto]
        #
        # The fourth rate constant is not as easy. JPL 15-10 pages 2-4 (396).
        #
        # a1 = ( k0*ki*[M] / ( ki + k0*[M] ) ) * 
        #        0.6 ** (-( 1+(log10(k0*[M]/ki))**2 )) cm^3 molec^-1 s^-1
        # k0 = 4.4E-32*(T/300)**-1.3
        # ki = 7.5E-11*(T/300)**-0.2
        #
        # With these reaction rate constants, the 
        #
        #

        ######################### Final Calculation ###########################

        #k3  = (8E-12   * np.exp( -2060 / self.D['lyT'] ))/100**3 # (m^3/molec)/s
        #k11 = (5.0E-12 * np.exp( 210 / self.D['lyT'] ))/100**3   # (m^3/molec)/s
        k3  = (8E-12   * np.exp( -2060 / self.lyT ))/100**3 # (m^3/molec)/s
        k11 = (5.0E-12 * np.exp( 210 / self.lyT ))/100**3   # (m^3/molec)/s

        #hk1 = 1.7E-12*np.exp( -940 / self.D['lyT'] )/100**3 #(m^3/molec)/s
        #hk2 = 1.0E-14*np.exp( -490 / self.D['lyT'] )/100**3 #(m^3/molec)/s


        source   = 2*self.j2*o2/self.nm*3600*24 # ppv / day
        sink_ox  = 2*k3*self.j3/(k2*o2)*3600*24 # ppv / day

        #sink_oh  = hk1*self.D['lyOH']
        #sink_ho2 = hk2*self.D['lyHO2']

        #a2 = 1.4E-10 * np.exp(-470/ self.D['lyT'] ) / 100**3
        #a5 = 1.8E-11 * np.exp( 180/ self.D['lyT'] ) / 100**3
        #a7 = 3.0E-11 * np.exp( 200/ self.D['lyT'] ) / 100**3
        a2 = 1.4E-10 * np.exp(-470/ self.lyT ) / 100**3
        a5 = 1.8E-11 * np.exp( 180/ self.lyT ) / 100**3
        a7 = 3.0E-11 * np.exp( 200/ self.lyT ) / 100**3

        #k0 = 4.4E-32*(self.D['lyT'] /300)**-1.3
        #ki = 7.5E-11*(self.D['lyT'] /300)**-0.2
        k0 = 4.4E-32*(self.lyT /300)**-1.3
        ki = 7.5E-11*(self.lyT /300)**-0.2
        a1 = ( k0*ki*self.nm/ ( ki + k0*self.nm) ) * \
               0.6 ** (-( 1+(np.log10(k0*self.nm/ki))**2 ))/100**3 # m^3 molec^-1 s^-1

        # HO2/OH = (a5*a1*[M][O2])/(a7*(a1*[M][O2]+a2*[O3]))
        #ho2_to_oh_ratio = a5*a1*o2/(a7*a1*o2+a2*self.D['lyO'])          # unitless
        ho2_to_oh_ratio = a5*a1*o2/(a7*a1*o2+a2*self.lyO)          # unitless
        # [H] = (a5*[O][OH])/(a1*[M][O2]+a2*[O3])
        #h_to_oh_ratio = (a5*o_to_o3_ratio*self.D['lyO'])/(a1*o2+a2*self.D['lyO'])
        h_to_oh_ratio = (a5*o_to_o3_ratio*self.lyO)/(a1*o2+a2*self.lyO)

        oh_to_hox_ratio = ( h_to_oh_ratio + 1 + ho2_to_oh_ratio)**-1  # unitless
        lyOH  = oh_to_hox_ratio * self.D['lyHOx']
        lyH = h_to_oh_ratio * lyOH
        lyHO2 = ho2_to_oh_ratio * lyOH
        #sink_h = a2 * lyH + a1 * lyH * o2 / self.D['lyO']
        sink_h = a2 * lyH + a1 * lyH * o2 / self.lyO
        sink_oh  = a5*lyOH
        sink_ho2 = a7*lyHO2

        sink_hox = (sink_h + (sink_oh+sink_ho2)*o_to_o3_ratio) \
            * self.nm* 3600*24 # ppv/day
        

        ####################### Chlorine Partitioning #########################

        #
        # The chemistry:
        #
        #   cl1:  ClO + O  ->  Cl  + O2   
        #   cl2:  Cl  + O3 ->  ClO + O2
        # 
        #   Assuming steady state, then Cl/ClO = (cl1*O)/(cl2*O3).
        #   Because we relate O to O3 with o_to_o3_ratio, the O3 concentration
        #   cancels and we're left with cl1*o_to_o3_ratio/cl2
        #
        # Rates from JPL document 15-10, pages 1-196:
        #
        #   cl1 = 2.8E-11*exp[85*T**-1]     cm^3/molec/s (T in K)
        #   cl2 = 2.3E-11*exp[-200*T**-1]   (the same as above)               
        #sink_clox = (sink_cl+sink_clo)*3600*24 # ppv/day

        #cl1 = 2.8E-11*np.exp(   85 / self.D['lyT'] )/100**3         # (m^3/molec)/s
        #cl2 = 2.3E-11*np.exp( -200 / self.D['lyT'] )/100**3         # (m^3/molec)/s
        cl1 = 2.8E-11*np.exp(   85 / self.lyT )/100**3         # (m^3/molec)/s
        cl2 = 2.3E-11*np.exp( -200 / self.lyT )/100**3         # (m^3/molec)/s

        cl_to_clo_ratio = cl1*o_to_o3_ratio/cl2           # unitless

        if len(np.where(o_to_o3_ratio==0)[0])>0:
          cl_to_clox_ratio = 0 
        else:
          cl_to_clox_ratio = ( 1 + 1/cl_to_clo_ratio)**-1  # unitless

        lyCl  = cl_to_clox_ratio * self.D['lyClOx']
        lyClO = (1-cl_to_clox_ratio) * self.D['lyClOx']
        
        #
        # The loss of Ox due to ClOx is:
        #
        # sink_clox = cl2*lyCl*lyO + cl1*lyClO*o_to_o3_ratio*lyO
        #
        # where lyO is the ozone mixing ratio. We'll avoid the application of
        # lyO until the lyO evolution line below, in case we want to apply an
        # internal ozone chemistry time integration routine later on. But to
        # adjust the units we'll also have to multiply by the number density of
        # air, which is self.D['nm'].
        #

        sink_clox = (cl2*lyCl + cl1*lyClO*o_to_o3_ratio)*self.nm*86400 # ppv/day

        #
        # I tried using an Adams-Bashforth fourth order timestepping scheme but
        # I did not notice a significant difference at equilibrium. Perhaps it 
        # takes longer to reach coupled equilibrium or maybe it's unstable under
        # certain conditions, though.
        #

        self.dlyO = self.D['timestepsize'] * \
                ( source\
                - sink_ox*self.lyO**2 \
                - sink_nox*self.lyO \
                - sink_hox*self.lyO \
                - sink_clox*self.lyO )
                                          
        #self.D['sink_hox'] = sink_hox*self.D['lyO']
        #self.D['source'] = source       #ppv / day
        #self.D['sink_ox'] = sink_ox*self.D['lyO']**2
        #self.D['sink_nox'] = sink_nox*self.D['lyO']
        #self.D['sink_clox'] = sink_clox*self.D['lyO']
        self.sink_hox = sink_hox*self.lyO
        self.source = source       #ppv / day
        self.sink_ox = sink_ox*self.lyO**2
        self.sink_nox = sink_nox*self.lyO
        self.sink_clox = sink_clox*self.lyO

    def record( self , finish = False ):

        if self.O is None: self.O = {}
        for key in self.D.keys():
            if self.extra[key][2]:

                output = self.D[key].copy()
                if type(output) is not np.ndarray:
                    output = np.array([output])

                if key in self.O.keys():
                    self.O[key] = np.append( self.O[key]            \
                                           , np.expand_dims(        \
                                             output                 \
                                           , axis = 0 )             \
                                           , axis = 0 )
                else:
                    self.O[key] = np.expand_dims( output, axis=0 )
            if finish: 
                #if key in self.O.keys(): 
                #    if key=='lyO':
                #      self.extra[key][0] = ('time', self.extra[key][0])
                #      self.O[key] = np.squeeze(self.O[key])
                #    else: 
                #      self.O[key] = np.mean(self.O[key],axis=0)
                #    #self.O[key] = np.mean(self.O[key],axis=0)
                #    print(key, self.extra[key], np.shape(self.O[key]))
                #else:
                #    self.O[key] = self.D[key]
                if key in self.O.keys(): 
                    if self.extra[key][0]=='constant':
                      self.extra[key][0] = 'time'
                    else:
                      self.extra[key][0] = ('time', self.extra[key][0])

                    self.O[key] = np.squeeze(self.O[key])
                else:
                    self.O[key] = self.D[key]


    def prep_output( self ):

        for key in self.O.keys():
            if self.extra[key][2]:
                self.O[key] = np.expand_dims( self.O[key] , 0 )

    def write_to_file( self , filename , init_file = False, \
            description = "RCE-PCE output" , verbose = False):

        if self.O is None: 
            raise Exception("You can't output until the model runs!")

        if filename[-3:] != '.nc': filename+='.nc' # Force nc file name

        nc_fid = Dataset( filename , 'w' )
        nc_fid.createDimension( "ly" , len(self.D['ly']) )
        nc_fid.createDimension( "lv" , len(self.D['lv']) )
        nc_fid.createDimension( "time" , len(self.D['time']))
        nc_fid.createDimension( "constant" , 1 )
        for key in np.sort(list(self.D.keys())):
            if verbose: print( key + "  " + str(self.extra[key] ))
            if self.extra[ key ][0] is not None:
                dims = (self.extra[ key ][0] )
            #else:
            #    dims = ("constant",)
            if hasattr( self.D[key] , 'dtype' ):
                datatype = self.D[key].dtype
            else:
                datatype = type(self.D[key])
            variable = nc_fid.createVariable( key , datatype , dims )
            if init_file:
                nc_fid.variables[ key ][:] = self.D[key]
            else:
              nc_fid.variables[ key ][:] = self.O[key]
            variable.form       = str(self.extra[key][0])
            variable.units      = str(self.extra[key][1])
            variable.recording  = str(self.extra[key][2])
        nc_fid.close()

    def set_transport_weights( self ):

        dx = ( self.D['lyZ'][1:  ] - self.D['lyZ'][ :-1] )*1000
        
        self.b00 = +1/dx
        self.bm1 = -1/dx

    def set_pressure( self ):
        self.lvP = self.D['lvP']
        self.lvP[1:] = self.D['lvP'][0]*np.exp(
                - 9.81 / 287.058 * np.cumsum( 
                    np.diff( self.D['lvZ'] * 1000 ) / self.lyT
                    )
                )
        
        #self.D['lyP'] = np.sqrt( self.D['lvP'][:-1]*self.D['lvP'][1:] )

    def lvt_consistency( self ):

        for iLy in range( 0 , len( self.D['lyT'] ) ):
            self.D['lvT'][iLy+1] = self.D['lyT'][iLy]*2 - self.D['lvT'][iLy]

    def convective_adjustment( self ):

        if not self.bool_run_rad: return

        ref = self.D['surface_temp']                                            \
            + self.D['crit_convec_lapse_rate'] * self.D['lyZ']
        if np.any(ref>self.D['lyT']): 
            self.D['cti'] = np.where( ref > self.D['lyT'] )[0][-1] #Convection top index
        self.D['lyT'] = np.max( [ self.D['lyT'] , ref ] , axis = 0 )

    def ozone_transport( self ):

        #dodz = np.diff( self.D['lyO'] ) / np.diff( self.D['lyZ']*1000 )
        #dodz = ( dodz[1:] + dodz[:-1] ) / 2
        #dodz = np.insert( dodz , 0 , 0 )
        #dodz = np.append( dodz , 0 )

        dfdx =    self.bm1 * self.D['lyO'][ :-1] \
                + self.b00 * self.D['lyO'][1:  ] \

        dfdx = np.insert( dfdx , 0 , 0 )

        #
        # This finite difference may be causing the wobbles in the ozone profile
        # and you can perhaps replace it with a difference that uses the i+1/2
        # and i - 1/2 points around each point i. These are the level values,
        # and they should probably be related to the average values in some
        # exponential terms.
        #

        self.lydo = -dfdx*self.lyU_lyO
        self.D['lyO'] += self.D['timestepsize'] * self.lydo

    def ozone_minimum( self ):

        self.D['lyO'][self.D['lyO']<self.D['ozone_min_val']] = \
                self.D['ozone_min_val']
        self.D['lyO'][0] = self.D['ozone_min_val']

    def set_water( self ):
        
        #
        # Given the same temperatures and pressures as a control run from the
        # MATLAB model, this calculation produces precisely the same water as
        # given in the control. I do not see a reason why this could be 
        # happenstance, so I conclude that this function behaves correctly.
        #

        A = self.D['lyT'].copy()*0.0
        A[self.D['lyT']>=273.15] = 17.62
        A[self.D['lyT']< 273.15] = 22.46

        B = self.D['lyT'].copy()*0.0
        B[self.D['lyT']>=273.15] = 243.12
        B[self.D['lyT']< 273.15] = 272.62

        SVP = 6.112*np.exp(
                A * ( 
                        ( self.D['lyT'] - 273.15 ) 
                        / 
                        ( self.D['lyT'] - 273.15 + B ) 
                    )
                ) # Saturation vapor pressure - hPa

        W = ( 0.5 * SVP ) / ( self.D['lyP']/100 - 0.5 * SVP )

        for iLy in range( 1 , self.D['nLayers'] ): 
            W[iLy] = min( W[iLy-1] , W[iLy] )

        W[self.D['cti']+1:] = W[self.D['cti']]*min( 0.9 , W[self.D['cti']] \
                        /W[self.D['cti']-1] ) \
                        ** np.arange(1,self.D['nLayers']-self.D['cti'])

        W[W<self.D['strat_wv']] = self.D['strat_wv']

        self.D['lyW'] = W

    def calculate_sza( self ):

        #
        # I calculated this in a different way, copying the code from the previ-
        # -ous model. However, I saw no difference, so I conclude that this is
        # just fine.
        #

        if self.D['eternal_day']<0: 
            time = self.D['day'] % 365 
        else:
            time = ( self.D['day'] % 1 ) + self.D['eternal_day']

        dec = -23.44/360*2*np.pi * np.cos( 2*np.pi*( time+10 )/365 )

        self.D['sza'] = np.arccos(
                 np.sin( self.D['lat']/360*2*np.pi ) # Sine Latitude
                *np.sin( dec ) # Sine Declination of Sun
                +
                +np.cos( self.D['lat']/360*2*np.pi ) # Cosine Latitude
                *np.cos( dec ) # Sine Declination of Sun
                *np.cos( ( (self.D['day'] % 1)-0.5 )*2*np.pi ) # Hour Angle
                ) * 360 / 2 / np.pi

        if np.isnan(self.D['sza']): raise Exception("sza calculation failed!")

    def set_bdc_temp_effect( self ):

        if not self.bool_run_rad: return

        #dtdz = np.diff( self.D['lyT'] ) / np.diff( self.D['lyZ']*1000 )
        #dtdz = ( dtdz[1:] + dtdz[:-1] ) / 2
        #dtdz = np.insert( dtdz , 0 , 0 )
        #dtdz = np.append( dtdz , 0 )

        # dtdz = np.diff(self.D['lvT'])/np.diff(self.D['lvZ']*1000)

        dfdx =    self.bm1 * self.D['lyT'][ :-1] \
                + self.b00 * self.D['lyT'][1:  ] \

        dfdx = np.insert( dfdx , 0 , 0 )

        #self.D['lyDH'] = - self.lyU_lyT*( dfdx + 9.81/1006) # K / day
        self.D['lyDH'] = - self.lyU_lyT*( dfdx + self.D['lyT']*(2/7)/7e3) # K / day
        
        
        # This derivative caused wobbles in the temperature profile
        #np.diff(self.D['lvT'])/np.diff(self.D['lvZ']*1000) \

        # I tried using a third order Adams-Bashforth to remove the wobbles,
        # but it didn't do anything noticeable by eye, so I went back to a
        # simple approximation.
        #self.D['lyDH'] = (1/12)*(   \
        #          23*self.lyDH_1    \
        #        - 16*self.lyDH_2    \
        #        +  5*self.lyDH_3 )

        #self.lyDH_3 = self.lyDH_2.copy()
        #self.lyDH_2 = self.lyDH_1.copy()

    def set_fdh_flags(self):
        self.bool_ozone_integration = False #turn off chem
        self.bool_temps_convection = False  #turn off convective adjustment

    def set_pce_only_flags(self):
        self.bool_ozone_integration = True  #turn on chem
        self.bool_run_rad = False           #turn off rad
        self.bool_temps_convection = False  #turn off convective adjustment

    def run_rad( self ):
        if self.D['eternal_day']<0: 
            jday = self.D['day'] % 365 
        else:
            jday = 0

        # Initialize input lines
        input_lines = []

        # Line for RRTM input file start
        input_lines.append( "$ Input from rce model\n" )
        
        # Line for RRTM options
        input_lines.append(                                                       
                 19*' '+'0'     #IAER     20                                      
                +29*' '+'0'     #IATM     50                                     
                +19*' '+'0'     #IXSECT   70                                    
                +12*' '+'1'     #ISCAT    83
                +'3'            #NUMANGS
                +'0'            #ISTRM    85  
                +'   98'        #IOUT     88-90 - edited rrtmg so 98 makes all 
                                # rrtmg_sw output and only broad lw output
                +' 0'           #IDRV     92                                      
                +' 0'           #IMCA     94                                     
                +'0'            #ICLD     95                                     
                +"   0"         #IDELM    99
                +"0"            #ICOS    100
                +'\n')
        
        #SW Data
        input_lines.append(
                 12*' '                                                         
                +('%3i' % jday ) #JULDAT                                          
                +3*' '                                                          
                +('%7.3f' % self.D['sza'] ) #SZA - 0 degrees is overhead
                +3*' '                                                          
                +('%2i' % -1 ) #ISOLVAR                                          
                +('%10.4f' % 0.0) #SCON - No solar variability or cycle           
                +('%10.5f' % 0.0) #SOLCYCFRAC - doesn't matter                  
                +140*' ' #SOLVAR                                                
                +'\n')

        # Lines for float parameters
        input_lines.append(                                                     
                 ('%10.3e' % self.D['surface_temp'])    #TBOUND                 
                +4*' '+'1'                              #IEMISS                 
                +2*' '+'1'                              #IREFLECT               
                +('%5.3f' % 0.8 ) #SW SEMISS
                +65*' '
                +('%5.3f' % 1.0 ) #LW SEMISS
                +'\n')

        # Flags for profiles
        input_lines.append(
                 ' 0'                                   #IFORM
                +( '%3i' % self.D['nLayers'] )          #NLAYERS
                +'\n')

        # Write layers
        i = 0
        input_lines.append(
                 ('%10.4f' % (self.D['lyP'][i]/100))                
                #+('%10.4f' %  self.D['lyT'][i])                    
                +('%10.4f' %  self.lyT[i])                    
                +' '*23
                +('%8.3f'  % (self.lvP[i]/100))                
                #+('%7.2f'  %  self.D['lvT'][i])                     
                +('%7.2f'  %  self.lvT[i])                     
                +' '*7
                +('%8.3f'  % (self.lvP[i+1]/100))              
                #+('%7.2f'  %  self.D['lvT'][i+1])                   
                +('%7.2f'  %  self.lvT[i+1])                   
                +'\n'                                               
                +('%10.3e' %  self.D['lyW'][i])                     
                +('%10.3e' %  self.D['lyCO2'][i])                   
                #+('%10.3e' %  self.D['lyO'][i])                     
                +('%10.3e' %  self.lyO[i])                     
                +('%10.3e' %  self.D['lyNO2'][i])                   
                +('%10.3e' %  self.D['lyCO'][i])                    
                +('%10.3e' %  self.D['lyCH4'][i])                   
                +('%10.3e' %  self.D['lyO2'][i])                    
                +('%10.3e' %  self.lyBroad[i])                  
                +'\n')
        for i in range(1,self.D['nLayers']): input_lines.append(    
                 ('%10.4f' % (self.D['lyP'][i]/100))                
                #+('%10.4f' %  self.D['lyT'][i])                     
                +('%10.4f' %  self.lyT[i])                     
                +' '*45                                             
                +('%8.3f'  % (self.lvP[i+1]/100))              
                #+('%7.2f'  %  self.D['lvT'][i+1])                   
                +('%7.2f'  %  self.lvT[i+1])                   
                +'\n'                                               
                +('%10.3e' %  self.D['lyW'][i])                     
                +('%10.3e' %  self.D['lyCO2'][i])                   
                #+('%10.3e' %  self.D['lyO'][i])                     
                +('%10.3e' %  self.lyO[i])                     
                +('%10.3e' %  self.D['lyNO2'][i])                   
                +('%10.3e' %  self.D['lyCO'][i])                    
                +('%10.3e' %  self.D['lyCH4'][i])                   
                +('%10.3e' %  self.D['lyO2'][i])                    
                +('%10.3e' %  self.lyBroad[i])                  
                +'\n')
        
        # Write the lines
        with open( 'INPUT_RRTM' , 'w' ) as f: f.writelines( input_lines )
        os.system( './rrtmg_lw' )
        lw_output = np.genfromtxt( 'OUTPUT_RRTM' 
                , skip_header=3 , skip_footer=8 )
        os.system( 'mv OUTPUT_RRTM OUTPUT_RRTM_LW' )

        # The variable, output, if iout is 0, is a 195x6
        # It has cols of: level, pressure, up flux, down flux, net flux, heating

        # Exclude the last heating rate because it's for the last layer of rrtm,
        # which is generated by rrtm to account for radiation changes from the
        # atmosphere above the top of the model.
        self.lyHL = lw_output[-1:0:-1,5] 

        # Run rrtmg_sw if sun is above horizon
        if abs(self.D['sza'])<90: 

            os.system( './rrtmg_sw' )
            sw_output = np.genfromtxt( 
                      'OUTPUT_RRTM' 
                    , dtype=float 
                    , usecols=(0,6,7) 
                    , skip_footer = 2738 )
            self.swo = sw_output
            os.system( 'mv OUTPUT_RRTM OUTPUT_RRTM_SW' )
            self.lyHS = sw_output[-1:0:-1,2][:194]

        else:
            self.lyHS = self.D['lyHL']*0.0


        if np.any(np.isnan( self.lyHL )): raise Exception("lyHL is nan!")
        if np.any(np.isnan( self.lyHS )): raise Exception("lyHS is nan!")

    def quasi_analytical_ozone( self ):

        #
        # This function sets output ozone to a guess based entirely on photoche-
        # -mistry. This should only be used to get a photochemical equilibruim
        # profile of ozone, It should be used by calling the wrapper function,
        # which should run for something like 100 days if also computiing tempe-
        # -rature changes and maybe just 10 days if not.
        #
        source = self.O['source']        #ppv / day
        sink_ox = self.O['sink_ox'] / self.O['lyO']**2
        sink_nox = self.O['sink_nox'] / self.O['lyO']
         
        guess = ( np.sqrt( sink_nox**2 \
                + 4 * source * sink_ox \
                ) - sink_nox ) / ( 2 * sink_ox )

        guess[guess<self.D['ozone_min_val']] = self.D['ozone_min_val']
        guess[guess>1.00] = 1.00

        guess = np.squeeze( guess )
        self.guess = guess

        self.O['lyO'] = guess
        self.ozone_minimum()

        if np.any( self.D['lyO'] < 0 ) \
                or np.any( self.D['lyO'] > 1 )      \
                or np.any( np.isnan(self.D['lyO'])) \
                or np.any( np.isinf(self.D['lyO']) ):
            raise Exception( "Quasi-Analytical method has bad ozone" )

    def set_mmair( self ):

        mmDry = 28.96 / 1000 # kg/mol
        mmH2O = 18.02 / 1000 # kg/mol
        r = ( ( 1 / self.D['lyW'] + 1 ) * mmDry/mmH2O - 1 ) ** -1
        self.D['molar_mass_air'] = mmDry * ( 1 - r ) + mmH2O * r

    def set_broad( self ):

        # This computes the variable WBOADL, which is the "broadening gases"
        # column density in each layer. The broadening gases are all those that
        # are not one of the otherwise radiatively active species in the rrdmg
        # code. That's why I use summol to remove all of the radiatively active
        # species.

        summol = self.D['lyCO2']    \
                +self.lyO      \
                +self.D['lyNO2']    \
                +self.D['lyCO']     \
                +self.D['lyCH4']    \
                +self.D['lyO2']     \
                +self.D['lyW']

        self.lyBroad = ( self.lvP[:-1]-self.lvP[1:] ) \
                / 9.81 / self.D['molar_mass_air'] \
                * 6.022E+23 / 100 / 100 * (1-summol) # molecules/cm^2
    
    def set_upwelling( self ):
        
        self.lyU_lyT = self.D['lyZ']*0.0 + self.D['strat_up_temps'] / 1000 * 86400
        self.lyU_lyO = self.D['lyZ']*0.0 + self.D['strat_up_ozone'] / 1000 * 86400
        
        if self.bool_impose_resw:
          
          current_qbo_time = self.computed_days % (28*30)
          i_currenttime = int(current_qbo_time)
          dtime = current_qbo_time - i_currenttime
          tanh_profile = 1-0.2*(np.tanh(self.D['lyZ']-20)-np.tanh(self.D['lyZ']-28))
          #resw units of m/day
          resw  = tanh_profile.squeeze()*((dtime*self.resw_in(i_time=i_currenttime+1)[:] \
              + (1-dtime)*self.resw_in(i_time=i_currenttime)[:])\
              / 1000 * 86400 ).squeeze()

          self.lyU_lyT += resw.squeeze()  #resw S term
          self.lyU_lyO += resw.squeeze()  #resw dO3/dz term


        if self.bool_temps_convection: 
          self.lyU_lyT[:self.D['cti']] = 0.0

    def update_experimental_variables( self ):

        for var in self.exp_strs.keys():
            exec("self.D['"+var+"']="+self.exp_strs[var])
        
    def assign_experimental_command( self, var, string ):

        self.exp_strs[var] = string

    def timestep( self ):

        # Set independent variables with varying values
        self.update_experimental_variables()

        # Check if we are in fdh mode
        if self.bool_fdh:
            self.set_fdh_flags()
        else:
            self.set_water()

        # Check if we are in PCE mode only
        if self.bool_pce_only:
            self.set_pce_only_flags()

        # Time adjustments
        self.calculate_sza()
        self.set_upwelling()
        self.set_mmair()

        # Molecular adjustments aside from ozone
        #self.set_pressure()
        #self.set_broad()
        #self.set_water()


        # Radiation
        if self.bool_run_rad:
            #run rad once with no T changes
            self.lyT = self.D['lyT_0']
            self.lvT = self.D['lvT_0']
            self.lyO = self.D['lyO']

            self.set_pressure()
            self.set_broad()
            self.run_rad()
            self.D['lyHS_T0']=self.lyHS
            self.D['lyHL_T0']=self.lyHL

            #run rad again with no O3 changes
            self.lyT = self.D['lyT']
            self.lvT = self.D['lvT']
            self.lyO = self.D['lyO_0']

            self.set_pressure()
            self.set_broad()
            self.run_rad()
            self.D['lyHS_O0']=self.lyHS
            self.D['lyHL_O0']=self.lyHL

            #run rad properly with both T and O3 changes
            self.lyT = self.D['lyT']
            self.lvt_consistency()
            self.lvT = self.D['lvT']
            self.lyO = self.D['lyO']

            self.set_pressure()
            self.set_broad()
            self.D['lyBroad'] = self.lyBroad
            self.D['lvP'] = self.lvP
            self.run_rad()
            self.D['lyHS']=self.lyHS
            self.D['lyHL']=self.lyHL
        else:
            self.D['lyHS']=self.D['lyHS']*0.0
            self.D['lyHL']=self.D['lyHL']*0.0
            self.D['lyDH']=self.D['lyDH']*0.0
        
        if self.bool_ozone_integration:
            ## Ozone changes
            if self.D['sza']<90:
                current_qbo_time = self.computed_days % (28*30)
                i_currenttime = int(current_qbo_time)
                dtime = current_qbo_time - i_currenttime
                #dnox ppv
                dnox  = (dtime*self.nox_in(i_time=i_currenttime+1)[:] \
                + (1-dtime)*self.nox_in(i_time=i_currenttime)[:])

                #run chemistry with T constant
                self.lyT = self.D['lyT_0']
                self.lvT = self.D['lvT_0']
                self.lyO = self.D['lyO']
                self.dnox = dnox.squeeze()
                self.set_pressure()
                self.ozone_chemistry()
                self.D['sink_hox_T0'] = self.sink_hox
                self.D['source_T0'] = self.source       #ppv / day
                self.D['sink_nox_T0'] = self.sink_nox
                self.D['sink_ox_T0'] = self.sink_ox
                self.D['sink_clox_T0'] = self.sink_clox

                #run chemistry with O3 constant
                self.lyT = self.D['lyT']
                self.lvT = self.D['lvT']
                self.lyO = self.D['lyO_0']
                self.dnox = dnox.squeeze()
                self.set_pressure()
                self.ozone_chemistry()
                self.D['sink_hox_O0'] = self.sink_hox
                self.D['source_O0'] = self.source       #ppv / day
                self.D['sink_nox_O0'] = self.sink_nox
                self.D['sink_ox_O0'] = self.sink_ox
                self.D['sink_clox_O0'] = self.sink_clox
                
                #run chemistry with NOx constant
                self.lyT = self.D['lyT']
                self.lvT = self.D['lvT']
                self.lyO = self.D['lyO']
                self.set_pressure()
                self.dnox = 0
                self.ozone_chemistry()
                self.D['sink_hox_N0'] = self.sink_hox
                self.D['source_N0'] = self.source       #ppv / day
                self.D['sink_nox_N0'] = self.sink_nox
                self.D['sink_ox_N0'] = self.sink_ox
                self.D['sink_clox_N0'] = self.sink_clox

                #run chemistry normally
                self.lyT = self.D['lyT']
                self.lvT = self.D['lvT']
                self.lyO = self.D['lyO']
                self.dnox = dnox.squeeze()
                self.set_pressure()
                self.ozone_chemistry()
                self.D['lyO'] += self.dlyO
                self.D['nm'] = self.nm
                self.D['j3'] = self.j3
                self.D['j2_hz'] = self.j2_hz
                self.D['j2_srb'] = self.j2_srb
                self.D['j2'] = self.j2
                self.D['lyNO2'] = self.lyNO2
                self.D['sink_hox'] = self.sink_hox
                self.D['source'] = self.source       #ppv / day
                self.D['sink_ox'] = self.sink_ox
                self.D['sink_nox'] = self.sink_nox
                self.D['sink_clox'] = self.sink_clox
            else:
                self.D['source']    = self.D['source']*0.0
                self.D['sink_ox']   = self.D['sink_ox']*0.0
                self.D['sink_nox']  = self.D['sink_nox']*0.0
                self.D['sink_hox']   = self.D['sink_hox']*0.0

                self.D['source_T0']    = self.D['source_T0']*0.0
                self.D['sink_ox_T0']   = self.D['sink_ox_T0']*0.0
                self.D['sink_nox_T0']  = self.D['sink_nox_T0']*0.0
                self.D['sink_hox_T0']   = self.D['sink_hox_T0']*0.0

                self.D['source_O0']    = self.D['source_O0']*0.0
                self.D['sink_ox_O0']   = self.D['sink_ox_O0']*0.0
                self.D['sink_nox_O0']  = self.D['sink_nox_O0']*0.0
                self.D['sink_hox_O0']   = self.D['sink_hox_O0']*0.0

                self.D['source_N0']    = self.D['source_N0']*0.0
                self.D['sink_ox_N0']   = self.D['sink_ox_N0']*0.0
                self.D['sink_nox_N0']  = self.D['sink_nox_N0']*0.0
                self.D['sink_hox_N0']   = self.D['sink_hox_N0']*0.0
            self.ozone_transport()
            self.ozone_minimum()

        # Temperature Changes
        #if self.bool_impose_qdyn:
        #  #self.D['lyDH'] = self.D['qdyn'] # K / day
        #  #fake QBO heating 
        #  Q1 = np.exp(-(self.D['lyZ']-35)**2/(2*20))*\
        #      np.cos(2*np.pi*self.D['time']/(28*30)+(self.D['lyZ']-35)/4)
        #  Q2 = 0.35*np.exp(-(self.D['lyZ']-24)**2/(2*20))*\
        #      np.sin(2*np.pi*self.D['time']/(28*30)+(self.D['lyZ']-35)/4)
        #  Q = 0.3*(Q1+Q2)
        #  self.D['lyDH'] = Q # K / day
        #else:
        self.set_bdc_temp_effect()

        if self.bool_fdh:
            #find the fdh tropopause
            fdh_trop = np.squeeze(np.where(self.D['lvP']<self.D['fdh_trop_p']*1e2))[0]
            #adjust temperature above this level only
            self.D['lyT'][fdh_trop:] += self.D['timestepsize']*(                               
                    self.D['lyHS'][fdh_trop:] 
                   +self.D['lyHL'][fdh_trop:] 
                   +self.D['lyDH'][fdh_trop:])                            
        elif self.bool_pce_only:
            # Run in PCE mode only. Do not let temperatures evolve
            self.D['lyT'] = self.I['lyT']
        else:
            self.D['lyT'] += self.D['timestepsize']*(                               
                    self.D['lyHS']+self.D['lyHL']+self.D['lyDH'] )                            

        if np.any(np.isnan( self.D['lyT'] )): raise Exception("Temp. is nan!")
        if self.bool_temps_convection: self.convective_adjustment()


    def wrapper( self , reset_initial=False ):

        """
        column.wrapper is the main function for running the column simulation.

        Parameters
        ----------
        reset_initial : boolean, optional
            If true, the initial (I) dictionay is populated with the output (O)
            dictionary. Only useful for use with the compare_plot plotting
            routine, which compares the I dictionary data to the O dictionary
            data. If reset_initial is used, the I dictionary will be therefore
            be the data used to initialize the specific wrapper call and not the
            data used to initialize the column object when it was created.
        
        """

        # Intialize a few things
        self.radtime    = 0
        self.runtime    = 0

        #self.prep_output()

        # add a time axis
        no_timesteps_to_record = int(self.D['recordtime']/self.D['timestepsize'])
        record_time = np.linspace(self.D['recordtime'], self.D['totaltime'],\
                      no_timesteps_to_record)
        self.add_variable('time', record_time, ['time', 'day', False])

        self.O = None

        self.ozone_chemistry_initialize()
        self.set_transport_weights()

        self.computed_days = 0
        timestep_number = int( self.D['totaltime'] / self.D['timestepsize'] )

        self.cpt = np.array([self.D['lyT'].min()])

        for timestepindex in range(0,timestep_number):
            clock = time.time()
            self.computed_days += self.D['timestepsize']

            # Set the day variable, which sets the day and the fraction of the
            # day that the run is on. Currently it's only allowed to use an
            # eternal day calculation, which means that time does not move
            # forward through days of the year.
            self.D['day'] += self.D['timestepsize']
            self.timestep()
            #if self.computed_days >= self.D['totaltime'] - self.D['recordtime']:
            #recordtime_startindex = timestep_number-1-no_timesteps_to_record
            recordtime_startindex = timestep_number-no_timesteps_to_record

            #if timestepindex >= timestep_number-1\
            #        -int(self.D['recordtime']/self.D['timestepsize']):
            if timestepindex >= recordtime_startindex:
                # Update time axis
                self.D['time'][timestepindex-recordtime_startindex] = self.D['day']
                self.record( finish = (timestepindex==timestep_number-1) )

            if self.bool_timing_output:

                self.runtime += time.time() - clock
                self.timer( timestep_number, timestepindex )

            self.cpt = np.append( self.cpt , self.D['lyT'].min() )

