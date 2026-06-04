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
        self.bool_impose_nox=False         # Impose dnox variation
        self.bool_pce_only=False            # Run PCE only. Read rad heating from input file and do not change.
                                            # (with transport of ozone)
        self.resw_in = pyg.open('resw.nc').resw 
        self.nox_in = pyg.open('nox.nc').dnox 
       
        self.chem_trop = 0
        self.next_night = 1
        self.count = 0  
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
        self.sigma_j3_o3 = np.array([\
        6.220000e+01, 5.760000e+01,
        5.260000e+01, 4.770000e+01, 4.290000e+01, 3.850000e+01,
        3.490000e+01, 3.240000e+01, 3.150000e+01, 3.260000e+01,
        3.630000e+01, 4.330000e+01, 5.390000e+01, 6.930000e+01,
        9.030000e+01, 1.180000e+02, 1.540000e+02, 1.990000e+02,
        2.550000e+02, 3.220000e+02, 4.010000e+02, 4.900000e+02,
        5.900000e+02, 6.930000e+02, 8.020000e+02, 9.080000e+02,
        1.001000e+03, 1.080000e+03, 1.125000e+03, 1.148000e+03,
        1.122000e+03, 1.064000e+03, 9.680000e+02, 8.400000e+02,
        6.980000e+02, 5.470000e+02, 4.060000e+02, 2.820000e+02,
        1.840000e+02, 1.130000e+02, 6.510000e+01, 3.543435e+01,
        1.930580e+01, 1.010784e+01, 5.190000e+00, 2.990000e+00,
        1.310000e+00, 6.970000e-01, 3.200000e-01, 1.460000e-01,
        7.790000e-02, 3.060000e-02, 1.360000e-02, 6.940000e-03,
        3.050000e-03, 1.300000e-03, 8.500000e-04, 5.720000e-04,
        5.420000e-04, 6.680000e-04, 9.560000e-04, 1.150000e-03,
        1.580000e-03, 2.580000e-03, 2.950000e-03, 3.930000e-03,
        6.560000e-03, 6.970000e-03, 8.820000e-03, 1.370000e-02,
        1.650000e-02, 1.850000e-02, 2.180000e-02, 3.660000e-02,
        3.670000e-02, 4.100000e-02, 4.810000e-02, 7.540000e-02,
        8.130000e-02, 8.160000e-02, 9.080000e-02, 1.210000e-01,
        1.600000e-01, 1.580000e-01, 1.660000e-01, 1.830000e-01,
        2.190000e-01, 2.670000e-01, 2.870000e-01, 2.950000e-01,
        3.190000e-01, 3.370000e-01, 3.580000e-01, 3.980000e-01,
        4.390000e-01, 4.670000e-01, 4.810000e-01, 4.640000e-01,
        4.460000e-01, 4.470000e-01, 4.760000e-01, 5.130000e-01,
        5.140000e-01, 4.780000e-01, 4.380000e-01, 4.060000e-01,
        3.820000e-01, 3.560000e-01, 3.270000e-01, 2.970000e-01,
        2.710000e-01, 2.510000e-01, 2.310000e-01, 2.100000e-01,
        1.900000e-01, 1.700000e-01, 1.510000e-01, 1.370000e-01,
        1.260000e-01, 1.130000e-01, 9.890000e-02, 8.680000e-02,
        7.840000e-02, 7.310000e-02, 6.960000e-02, 6.220000e-02,
        5.430000e-02, 4.780000e-02, 4.420000e-02, 4.320000e-02,
        4.470000e-02, 4.250000e-02, 3.380000e-02, 2.860000e-02,
        2.620000e-02, 2.600000e-02, 2.940000e-02, 3.180000e-02,
        2.620000e-02, 2.080000e-02, 1.730000e-02, 1.570000e-02,
        1.560000e-02, 1.860000e-02, 2.210000e-02, 2.060000e-02])*1E-20*1E-4 #m^2

        self.flux_j3=np.array([
        3.62e+15, 4.73e+15, 5.61e+15, 6.63e+15, 6.90e+15, 9.56e+15,
        1.15e+16, 1.27e+16, 1.52e+16, 1.78e+16, 2.20e+16, 2.69e+16,
        4.54e+16, 7.14e+16, 8.35e+16, 8.39e+16, 1.08e+17, 1.18e+17,
        1.60e+17, 1.34e+17, 1.41e+17, 1.57e+17, 1.38e+17, 1.60e+17,
        1.45e+17, 2.20e+17, 1.99e+17, 1.97e+17, 1.94e+17, 2.91e+17,
        4.95e+17, 4.53e+17, 1.07e+18, 1.20e+18, 1.10e+18, 1.04e+18,
        8.24e+17, 1.52e+18, 2.15e+18, 3.48e+18, 3.40e+18, 3.22e+18,
        4.23e+18, 4.95e+18, 5.44e+18, 5.93e+18, 6.95e+18, 8.15e+18,
        7.81e+18, 8.35e+18, 8.14e+18, 8.53e+18, 9.17e+18, 8.38e+18,
        1.04e+19, 1.10e+19, 9.79e+18, 1.13e+19, 8.89e+18, 1.14e+19,
        9.17e+18, 1.69e+19, 1.70e+19, 1.84e+19, 1.87e+19, 1.95e+19,
        1.81e+19, 1.67e+19, 1.98e+19, 2.02e+19, 2.18e+19, 2.36e+19,
        2.31e+19, 2.39e+19, 2.38e+19, 2.39e+19, 2.44e+19, 2.51e+19,
        2.30e+19, 2.39e+19, 2.48e+19, 2.40e+19, 2.46e+19, 2.49e+19,
        2.32e+19, 2.39e+19, 2.42e+19, 2.55e+19, 2.51e+19, 2.49e+19,
        2.55e+19, 2.53e+19, 2.54e+19, 2.50e+19, 2.57e+19, 2.58e+19,
        2.67e+19, 2.67e+19, 2.70e+19, 2.62e+19, 2.69e+19, 2.63e+19,
        2.68e+19, 2.66e+19, 2.59e+19, 2.69e+19, 2.61e+19, 2.62e+19,
        2.62e+19, 2.63e+19, 2.60e+19, 2.55e+19, 2.48e+19, 2.57e+19,
        2.61e+19, 2.61e+19, 2.62e+19, 2.62e+19, 2.57e+19, 2.52e+19,
        2.60e+19, 2.58e+19, 2.52e+19, 2.51e+19, 2.48e+19, 2.45e+19,
        2.48e+19, 2.45e+19, 2.44e+19, 2.39e+19, 2.40e+19, 2.41e+19,
        2.40e+19, 2.38e+19, 2.34e+19, 2.32e+19, 2.30e+19, 2.33e+19,
        2.34e+19, 2.29e+19, 2.29e+19, 2.27e+19, 2.27e+19, 2.20e+19,
        2.22e+19, 2.18e+19])

        self.quantum_yield_o1d = np.array([
        0.48, 0.51, 0.54, 0.57, 0.59, 0.62,
        0.65, 0.67, 0.72, 0.74, 0.77, 0.80,
        0.83, 0.86, 0.90, 0.90, 0.90, 0.90,
        0.90, 0.90, 0.90, 0.90, 0.90, 0.90,
        0.90, 0.90, 0.90, 0.90, 0.90, 0.90,
        0.90, 0.90, 0.90, 0.90, 0.90, 0.90,
        0.90, 0.90, 0.9]) 

        self.flux_j3 = self.add_ly_dim( self.flux_j3 )
        self.sigma_j3_o3 = self.add_ly_dim( self.sigma_j3_o3 )
        self.quantum_yield_o1d = self.add_ly_dim( self.quantum_yield_o1d )
        
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
    
    def hno3_chemistry_initialize( self ):
        self.sigma_hno3 = np.array([
            1.14430250e+03, 1.00769625e+03, 8.50835000e+02, 6.78090000e+02,
            5.16795000e+02, 3.84465500e+02, 2.73001000e+02, 1.83768750e+02,
            1.18833000e+02, 7.39605000e+01, 4.49620000e+01, 2.81765000e+01,
            1.85069750e+01, 1.33484500e+01, 1.01898025e+01, 8.02318000e+00,
            6.51023500e+00, 5.21254000e+00, 4.16701000e+00, 3.21068250e+00,
            2.66233750e+00, 2.30103000e+00, 2.08960000e+00, 1.99314500e+00,
            1.96208750e+00, 1.95000000e+00, 1.92925000e+00, 1.88102000e+00,
            1.80262500e+00, 1.68126000e+00, 1.52453250e+00, 1.33684750e+00,
            1.13463000e+00, 9.24049000e-01, 7.19559500e-01, 5.32560250e-01,
            3.70843750e-01, 2.41866250e-01, 1.43863000e-01, 8.05072000e-02,
            4.15000000e-02, 1.97000000e-02, 9.50000000e-03, 4.31000000e-03,
            2.19500000e-03, 1.03000000e-03, 5.95000000e-04, 4.20000000e-04
            ])*1e-20*1e-4# m^2
        
        self.sigma_hno3_o3 = np.array([
            4.290000e+01, 3.850000e+01, 3.490000e+01, 3.240000e+01, 3.150000e+01,
            3.260000e+01, 3.630000e+01, 4.330000e+01, 5.390000e+01, 6.930000e+01,
            9.030000e+01, 1.180000e+02, 1.540000e+02, 1.990000e+02, 2.550000e+02,
            3.220000e+02, 4.010000e+02, 4.900000e+02, 5.900000e+02, 6.930000e+02,
            8.020000e+02, 9.080000e+02, 1.001000e+03, 1.080000e+03, 1.125000e+03,
            1.148000e+03, 1.122000e+03, 1.064000e+03, 9.680000e+02, 8.400000e+02,
            6.980000e+02, 5.470000e+02, 4.060000e+02, 2.820000e+02, 1.840000e+02,
            1.130000e+02, 6.510000e+01, 3.543435e+01, 1.930580e+01, 1.010784e+01,
            5.190000e+00, 2.990000e+00, 1.310000e+00, 6.970000e-01, 3.200000e-01,
            1.460000e-01, 7.790000e-02, 3.060000e-02])*1e-20*1e-4# m^2

        self.B_coeff_hno3 = np.array([
            0.00,  0.95,  1.67,  1.65,  1.67,  1.71,
            1.75,  1.82,  1.93,  2.06,  2.16,  2.17,
            2.19,  2.10,  1.98,  1.84,  1.78,  1.83,
            1.89,  1.97,  1.97,  1.85,  1.65,  1.41,
            1.25,  1.16,  1.12,  1.14,  1.19,  1.29,
            1.49,  1.63,  1.76,  1.92,  2.13,  2.37,
            2.73,  3.15,  3.69,  4.25,  5.20,  6.45,
            7.35,  9.75, 10.05, 11.80, 10.70,  9.30]) * 1e-3 #K-1
        
        self.flux_hno3=np.array([
            6.90e+15, 9.56e+15, 1.15e+16, 1.27e+16, 1.52e+16, 1.78e+16, 
            2.20e+16, 2.69e+16, 4.54e+16, 7.14e+16, 8.35e+16, 8.39e+16, 
            1.08e+17, 1.18e+17, 1.60e+17, 1.34e+17, 1.41e+17, 1.57e+17, 
            1.38e+17, 1.60e+17, 1.45e+17, 2.20e+17, 1.99e+17, 1.97e+17, 
            1.94e+17, 2.91e+17, 4.95e+17, 4.53e+17, 1.07e+18, 1.20e+18, 
            1.10e+18, 1.04e+18, 8.24e+17, 1.52e+18, 2.15e+18, 3.48e+18, 
            3.40e+18, 3.22e+18, 4.23e+18, 4.95e+18, 5.44e+18, 5.93e+18, 
            6.95e+18, 8.15e+18, 7.81e+18, 8.35e+18, 8.14e+18, 8.53e+18
            ])# photons / m^2 / s 
 
        self.flux_hno3 = self.add_ly_dim( self.flux_hno3 )
        self.sigma_hno3 = self.add_ly_dim( self.sigma_hno3 )
        self.sigma_hno3_o3 = self.add_ly_dim( self.sigma_hno3_o3 )
        self.B_coeff_hno3 = self.add_ly_dim( self.B_coeff_hno3 )
    
    def n2o_chemistry_initialize( self ):
        self.sigma_n2o = np.array([
          1.46540495e+01,
          1.45874895e+01, 1.42010253e+01, 1.35058202e+01, 1.25354719e+01,
          1.13432482e+01, 9.99683329e+00, 8.57253701e+00, 7.14669642e+00,
          5.78685613e+00, 4.54763342e+00, 3.46638875e+00, 2.56113451e+00,
          1.83353770e+00, 1.27179449e+00, 8.54623304e-01, 5.56458822e-01,
          3.51306810e-01, 2.15282509e-01, 1.28228279e-01, 7.43747285e-02,
          4.21179713e-02, 2.33619403e-02, 1.27455354e-02, 6.87351135e-03,
          3.68588018e-03, 1.97962293e-03, 1.07391945e-03, 5.94409397e-04])*1e-20*1e-4# m^2
        
        self.sigma_n2o_o3 = np.array([
           62.2,   62.2,   62.2,   62.2,
           57.6,   52.6,   47.7,   42.9,   38.5,   34.9,   32.4,  
           31.5,   32.6,   36.3,   43.3,   53.9,   69.3,   90.3, 
          118.0,  154.0,  199.0,  255.0,  322.0,  401.0,  490.0, 
          590.0,  693.0,  802.0,  908.0])*1e-20*1e-4# m^2  

        self.flux_n2o=np.array([
          3.04e+15, 3.19e+15, 2.93e+15, 3.62e+15, 
          4.73e+15, 5.61e+15, 6.63e+15, 6.90e+15, 9.56e+15, 1.15e+16, 1.27e+16, 
          1.52e+16, 1.78e+16, 2.20e+16, 2.69e+16, 4.54e+16, 7.14e+16, 8.35e+16, 
          8.39e+16, 1.08e+17, 1.18e+17, 1.60e+17, 1.34e+17, 1.41e+17, 1.57e+17, 
          1.38e+17, 1.60e+17, 1.45e+17, 2.20e+17])# photons / m^2 / s 
        
        self.flux_n2o = self.add_ly_dim( self.flux_n2o )
        self.sigma_n2o = self.add_ly_dim( self.sigma_n2o )
        self.sigma_n2o_o3 = self.add_ly_dim( self.sigma_n2o_o3 )
    
    def no2_chemistry_initialize( self ):
        self.sigma_no2 = np.array([
          13.2, 16.0,  18.5, 20.8, 24.2, 27.2, 29.4, 33.0, 37.0,
          38.6, 43.5, 47.7, 49.2, 53.7, 55.2, 58.4, 58.5, 59.2, 
          62.4, 58.5])*1e-20*1e-4# m^2
        
        self.sigma_no2_o3 = np.array([
          3.543435e+01, 1.930580e+01, 1.010784e+01,
          5.190000e+00, 2.990000e+00, 1.310000e+00, 6.970000e-01, 3.200000e-01,
          1.460000e-01, 7.790000e-02, 3.060000e-02, 1.360000e-02, 6.940000e-03,
          3.050000e-03, 1.300000e-03, 8.500000e-04, 5.720000e-04, 5.420000e-04,
          6.680000e-04, 9.560000e-04])*1e-20*1e-4# m^2  

        self.flux_no2 = np.array([
          3.22e+18, 4.23e+18, 4.95e+18, 5.44e+18, 
          5.93e+18, 6.95e+18, 8.15e+18, 7.81e+18, 8.35e+18, 8.14e+18, 8.53e+18, 
          9.17e+18, 8.38e+18, 1.04e+19, 1.10e+19, 9.79e+18, 1.13e+19, 8.89e+18, 
          1.14e+19, 9.17e+18])# photons / m^2 / s 
        
        self.flux_no2 = self.add_ly_dim( self.flux_no2 )
        self.sigma_no2 = self.add_ly_dim( self.sigma_no2 )
        self.sigma_no2_o3 = self.add_ly_dim( self.sigma_no2_o3 )
    
    def n2o5_chemistry_initialize( self ):
        
        self.sigma_n2o5 = np.array([
          7.44311000e+02, 6.83986075e+02, 6.12456450e+02, 5.27777112e+02,
          4.33458562e+02, 3.38744550e+02, 2.75956750e+02, 2.22505563e+02,
          1.77295338e+02, 1.41087250e+02, 1.14600825e+02, 9.60772000e+01,
          8.11300350e+01, 7.09726413e+01, 6.31044250e+01, 5.74622525e+01,
          5.17898200e+01, 4.44357050e+01, 3.96909200e+01, 3.49740575e+01,
          3.07250563e+01, 2.68063437e+01, 2.33016875e+01, 2.01267675e+01,
          1.72757187e+01, 1.48114200e+01, 1.26030350e+01, 1.06087863e+01,
          9.02674500e+00, 7.53665250e+00, 6.18777475e+00, 5.00167200e+00,
          3.97242187e+00, 3.10458887e+00, 2.38461975e+00, 1.80111600e+00,
          1.34725000e+00, 1.00300000e+00, 7.46300000e-01, 5.55900000e-01,
          4.16500000e-01, 3.12800000e-01, 2.35875000e-01, 1.78500000e-01,
          1.34300000e-01, 1.02000000e-01, 7.69250000e-02, 5.82250000e-02,
          4.37325000e-02, 3.25550000e-02, 2.47775000e-02, 1.92100000e-02,
          1.47475000e-02, 1.14750000e-02, 9.13750000e-03, 6.80000000e-03,
          5.52500000e-03, 4.25000000e-03])*1e-20*1e-4# m^2


        self.sigma_n2o5_o3 = np.array([
          3.150000e+01,  3.260000e+01,  3.630000e+01,  4.330000e+01,  5.390000e+01, 
          6.930000e+01,  9.030000e+01,  1.180000e+02,  1.540000e+02,  1.990000e+02, 
          2.550000e+02,  3.220000e+02,  4.010000e+02,  4.900000e+02,  5.900000e+02, 
          6.930000e+02,  8.020000e+02,  9.080000e+02,  1.001000e+03,  1.080000e+03, 
          1.125000e+03,  1.148000e+03,  1.122000e+03,  1.064000e+03,  9.680000e+02, 
          8.400000e+02,  6.980000e+02,  5.470000e+02,  4.060000e+02,  2.820000e+02, 
          1.840000e+02,  1.130000e+02,  6.510000e+01,  3.543435e+01,  1.930580e+01, 
          1.010784e+01,  5.190000e+00,  2.990000e+00,  1.310000e+00,  6.970000e-01, 
          3.200000e-01,  1.460000e-01,  7.790000e-02,  3.060000e-02,  1.360000e-02, 
          6.940000e-03,  3.050000e-03,  1.300000e-03,  8.500000e-04,  5.720000e-04, 
          5.420000e-04,  6.680000e-04,  9.560000e-04,  1.150000e-03,  1.580000e-03, 
          2.580000e-03,  2.950000e-03,  3.930000e-03])*1e-20*1e-4# m^2  

        self.flux_n2o5 = np.array([
          1.52e+16, 1.78e+16, 2.20e+16, 2.69e+16, 4.54e+16, 7.14e+16, 8.35e+16, 8.39e+16,
          1.08e+17, 1.18e+17, 1.60e+17, 1.34e+17, 1.41e+17, 1.57e+17, 1.38e+17, 1.60e+17,
          1.45e+17, 2.20e+17, 1.99e+17, 1.97e+17, 1.94e+17, 2.91e+17, 4.95e+17, 4.53e+17,
          1.07e+18, 1.20e+18, 1.10e+18, 1.04e+18, 8.24e+17, 1.52e+18, 2.15e+18, 3.48e+18,
          3.40e+18, 3.22e+18, 4.23e+18, 4.95e+18, 5.44e+18, 5.93e+18, 6.95e+18, 8.15e+18,
          7.81e+18, 8.35e+18, 8.14e+18, 8.53e+18, 9.17e+18, 8.38e+18, 1.04e+19, 1.10e+19,
          9.79e+18, 1.13e+19, 8.89e+18, 1.14e+19, 9.17e+18, 1.69e+19, 1.70e+19, 1.84e+19,
          1.87e+19, 1.95e+19])# photons / m^2 / s     
        
        self.flux_n2o5 = self.add_ly_dim( self.flux_n2o5 )
        self.sigma_n2o5 = self.add_ly_dim( self.sigma_n2o5 )
        self.sigma_n2o5_o3 = self.add_ly_dim( self.sigma_n2o5_o3 )
    
    def clono2_chemistry_initialize( self ):
        self.sigma_clono2 = np.array([
          301.60,    287.94,    279.47,  278.576,   284.53,  295.55,
          310.00,    326.34,    338.55,   344.55,   338.36,  322.99,
          297.08,    264.05,    227.18,   192.12,   158.88,  131.74,
          108.52,     89.58,     74.44,    60.93,    51.41,   43.60,
           37.07,     31.48,     26.61,    22.18,    18.38,   15.00])*1e-20*1e-4# m^2

        self.sigma_clono2_o3 = np.array([
          34.9,  32.4,  31.5,   32.6,   36.3,   43.3,  53.9,   69.3,   90.3,  118,
           154,   199,   255,    322,    401,    490,   590,    693,   802,   908,
          1001,  1080,  1125,   1148,   1122,   1064,   968,    840,   698,   547])*1e-20*1e-4# m^2  

        self.flux_clono2 = np.array([
          1.15e+16, 1.27e+16, 1.52e+16, 1.78e+16, 2.20e+16, 2.69e+16, 4.54e+16, 7.14e+16,
          8.35e+16, 8.39e+16, 1.08e+17, 1.18e+17, 1.60e+17, 1.34e+17, 1.41e+17, 1.57e+17,
          1.38e+17, 1.60e+17, 1.45e+17, 2.20e+17, 1.99e+17, 1.97e+17, 1.94e+17, 2.91e+17,
          4.95e+17, 4.53e+17, 1.07e+18, 1.20e+18, 1.10e+18, 1.04e+18])# photons / m^2 / s     
        
        self.flux_clono2 = self.add_ly_dim( self.flux_clono2 )
        self.sigma_clono2 = self.add_ly_dim( self.sigma_clono2 )
        self.sigma_clono2_o3 = self.add_ly_dim( self.sigma_clono2_o3 )

    def reaction_rates( self ):

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
        o3_col = nm_col * self.lyO
        hno3_col = nm_col * self.D['lyHNO3']
        n2o_col = nm_col * self.D['lyN2O']
        no2_col = nm_col * self.D['lyNO2']
        n2o5_col = nm_col * self.D['lyN2O5']
        clono2_col = nm_col * self.D['lyClONO2']

        
        # over-model-top oxygen and ozone columns
        col_o2_toa = (self.D['lvP'][-1]*6.022E+23) \
                / (9.81*self.D['lyO2'][-1]*0.032) # molec/m^2
        # total overhead column densities ( molecules / meter^2 )
        o2_tcol = cosfac * ( np.flipud( np.cumsum( np.flipud( o2_col ) ) ) + \
                col_o2_toa )
        o3_tcol = cosfac * ( np.flipud( np.cumsum( np.flipud( o3_col ) ) ) + \
                self.col_o3_toa )
        hno3_tcol = cosfac * ( np.flipud( np.cumsum( np.flipud( hno3_col))))
        n2o_tcol = cosfac * ( np.flipud( np.cumsum( np.flipud( n2o_col ))))
        no2_tcol = cosfac * ( np.flipud( np.cumsum( np.flipud( no2_col ))))
        n2o5_tcol = cosfac * ( np.flipud( np.cumsum( np.flipud( n2o5_col ))))
        clono2_tcol = cosfac * ( np.flipud( np.cumsum( np.flipud( clono2_col ))))

        ############################### Radiation ##############################
        # J3 calculation
        o3_tr = np.exp( - o3_tcol * self.sigma_j3_o3 )      # You can consolidate this
        self.j3 = np.sum( o3_tr   \
            * self.flux_j3 * self.sigma_j3_o3 , axis=0 ) 

        # J3* calculation wavelength 193 to 305nm
        o3_tre = np.exp( - o3_tcol * self.sigma_j3_o3[10:49] )      
        self.j3e = np.sum( self.quantum_yield_o1d * o3_tre \
            * self.flux_j3[10:49,:] * self.sigma_j3_o3[10:49,:], axis=0 ) 

        # JHNO3 calculation
        #σ(λ, T) = σ(λ, 298 K) exp (B(λ) (T – 298)), T in K
        self.sigma_hno3_eff = self.sigma_hno3  \
            * np.exp(self.B_coeff_hno3*(self.lyT-298))
        hno3_tr = np.exp( - hno3_tcol * self.sigma_hno3_eff ) \
                * np.exp( - o3_tcol   * self.sigma_hno3_o3 )
        self.jhno3 = np.sum( hno3_tr \
            * self.flux_hno3 * self.sigma_hno3_eff , axis=0 ) 
        
        # JN2O calculation
        n2o_tr = np.exp( - n2o_tcol * self.sigma_n2o ) \
               * np.exp( - o3_tcol  * self.sigma_n2o_o3 )

        # Add a fake factor here to correct n2o
        self.jn2o = 0.1*np.sum( n2o_tr * self.flux_n2o * self.sigma_n2o, axis=0 ) 
        
        # JNO2 calculation
        no2_tr = np.exp( - no2_tcol * self.sigma_no2 ) \
               * np.exp( - o3_tcol  * self.sigma_no2_o3 )
        self.jno2 = np.sum( no2_tr * self.flux_no2 * self.sigma_no2, axis=0 ) 

        # JN2O5 calculation
        n2o5_tr = np.exp( - n2o5_tcol * self.sigma_n2o5 ) \
                * np.exp( - o3_tcol   * self.sigma_n2o5_o3 )
        self.jn2o5 = np.sum( n2o5_tr * self.flux_n2o5 * self.sigma_n2o5, axis=0 ) 
        
        # JClONO2 calculation
        clono2_tr = np.exp( - clono2_tcol * self.sigma_clono2 ) \
                * np.exp( - o3_tcol   * self.sigma_clono2_o3 )
        self.jclono2 = np.sum( clono2_tr * self.flux_clono2 * self.sigma_clono2, axis=0 ) 
        
        # J2 calculation 
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
    
    def noy_chemistry( self ):

        ################### N2O ##############################
        # N2O for the NOx budget part of the chemistry:
        #
        # There are two main sinks of N2O
        #   jn2o (90%):   N2O + hv    ->  N2 + O2  Primary sink of N2O (90%)
        #  k_n2o (10%):   N2O + O(1D) ->  2NO       Branching ratio=0.61. Source of 2 NO
        #             :   N2O + O(1D) ->  N2 + O2   Branching ratio=0.39

        #O(1D) to O3 ratio
        # O(1D)_to_O3 = j3e/(k_n2 N2 + k_o2 O2 + k_h2o H2O + k_n2o N2O)
        k_n2 = 2.15E-11 * np.exp(110/self.lyT)/(100**3)         # (m^3/molec)/s
        k_o2 = 3.3E-11 * np.exp(55/self.lyT)/(100**3)         # (m^3/molec)/s
        k_h2o = 1.63E-10 * np.exp(60/self.lyT)/(100**3)         # (m^3/molec)/s
        k_n2o = 1.19E-10 * np.exp(20/self.lyT)/(100**3)         # (m^3/molec)/s
        
        o2 = self.nm*self.D['lyO2']
        n2 = self.nm*0.78           
        h2o = self.nm*self.D['lyW']
        lyN2O = self.D['lyN2O']
        
        self.oe_to_o3_ratio = self.j3e / \
            (k_n2*n2 + k_o2*o2 + k_h2o*h2o + k_n2o* self.nm * lyN2O) # Quenching from N2,O2,H2O and sink from N2O
        self.odd_oe =  self.oe_to_o3_ratio * self.lyO * self.nm           #density of O(1D)

        sink_n2o = (0.9*self.jn2o + 0.1* k_n2o*self.odd_oe)*3600*24 # ppv / day

        self.dlyN2O = - self.D['timestepsize']*sink_n2o*lyN2O
        self.sink_n2o = sink_n2o*lyN2O

        ############ NOy = NOx + HNO3 + N2O5 + ClONO2 ####################
        #  NOy is transported as a family
        #  NOy is produced by (slow):
        #  k_n2o:   N2O + O  ->  2NO       Branching ratio=0.61. Source of 2 NO

        lyNOy = self.D['lyNOy']
        lyN2O5 = self.D['lyN2O5']
        lyClONO2 = self.D['lyClONO2']

        # Source NO from N2O + O 
        # Source of NO2 from photolysis of N2O5 (does not change NOy)
        # jn2o5: N2O5 + hv -> NO2 + NO3
        # NO3 rapidly gets turned into NO2 (timescale of s)
        #         NO3 + hv -> NO2 + O
        
        sink_n2o5 = self.jn2o5*lyN2O5
        self.dlyN2O5 = self.D['timestepsize']*(-sink_n2o5)*3600*24 # ppv / day
        
        # Source of NO2 from photolysis of ClONO2 (does not change NOy)
        # jclono2: ClONO2 + hv -> ClO + NO2

        sink_clono2 = self.jclono2*lyClONO2
        self.dlyClONO2 = self.D['timestepsize']*(-sink_clono2)*3600*24 # ppv / day

        # add these sources to nox
        source_nox = 2*0.61*k_n2o*lyN2O*self.odd_oe + 2*sink_n2o5 + sink_clono2
        self.dlyNOx = self.D['timestepsize']*(source_nox)*3600*24 # ppv / day
       
        # Parametrise general removal of noy from atmosphere ?
        #jnoy = 1e-9*np.exp((self.D['lyZ'] - 40)/2)
        #jnoy = 1e-3*np.exp((self.D['lyZ']/4 - 45)/2)
        jnoy = 1e-7
        sink_noy = jnoy*lyNOy
        self.dlyNOy = self.D['timestepsize']*(-sink_noy \
            +2*0.61*k_n2o*lyN2O*self.odd_oe )*3600*24 # ppv / day
        
        

    def termolecular_k_calc( self, k_0, k_inf, nm):
        return ((k_inf * k_0 * nm)/(k_inf + k_0 * nm))\
              * 0.6**(1/(1+(np.log10(k_0*nm/k_inf))**2))# (m^3/molec)^2/s

    def ozone_chemistry( self ):
        
        ######################### Oxygen Net Change ###########################
        #
        # From B&S page 273-277 (283-287)
        #
        #   J2: O2 + hv ->  2O
        #   k1: 2O + M  ->  O2 + M              (only relevant in thermosphere)
        #   k3: O3 + O  -> 2O2
        #  J3g: O3 + hv -> O2(3 sigma g) + O(3P)
        #  J3e: O3 + hv -> O2(1 delta g) + O(1D)
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
        #   O(1D)/O3 ~ J3e*/(k4a*[N2]+k4b*[O2])
        #
        # A ratio for O(3P) to O3 is also given.
        #
        #   O(3P)/O3 ~ J3/(k2*[M2]*[O2])
        #
        o2 = self.nm*self.D['lyO2']
        k3  = (8E-12   * np.exp( -2060 / self.lyT ))/100**3 # (m^3/molec)/s
        #k11 = (5.0E-12 * np.exp( 210 / self.lyT ))/100**3   # (m^3/molec)/s
        
        k2  = (6E-34   * ( self.lyT / 300 )**(-2.4))/100**6 # (m^3/molec)^2/s
        self.o_to_o3_ratio = self.j3 / (k2*o2*self.nm) # O/O3
        
        # k3 O + O3 -> 2 O2

        source   = 2*self.j2*o2/self.nm*3600*24 # ppv / day
        sink_ox  = 2*k3*self.j3/(k2*o2)*3600*24 # ppv / day


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

        a2 = 1.4E-10 * np.exp(-470/ self.lyT ) / 100**3
        a5 = 1.8E-11 * np.exp( 180/ self.lyT ) / 100**3
        a7 = 3.0E-11 * np.exp( 200/ self.lyT ) / 100**3

        k0 = 4.4E-32*(self.lyT /300)**-1.3
        ki = 7.5E-11*(self.lyT /300)**-0.2
        a1 = ( k0*ki*self.nm/ ( ki + k0*self.nm) ) * \
               0.6 ** (-( 1+(np.log10(k0*self.nm/ki))**2 ))/100**3 # m^3 molec^-1 s^-1

        # HO2/OH = (a5*a1*[M][O2])/(a7*(a1*[M][O2]+a2*[O3]))
        ho2_to_oh_ratio = a5*a1*o2/(a7*a1*o2+a2*self.lyO)          # unitless
        # [H] = (a5*[O][OH])/(a1*[M][O2]+a2*[O3])
        h_to_oh_ratio = (a5*self.o_to_o3_ratio*self.lyO)/(a1*o2+a2*self.lyO)

        oh_to_hox_ratio = ( h_to_oh_ratio + 1 + ho2_to_oh_ratio)**-1  # unitless
        self.lyOH  = oh_to_hox_ratio * self.D['lyHOx']
        lyH = h_to_oh_ratio * self.lyOH
        lyHO2 = ho2_to_oh_ratio * self.lyOH
        sink_h = a2 * lyH + a1 * lyH * o2 / self.lyO
        sink_oh  = a5*self.lyOH
        sink_ho2 = a7*lyHO2

        sink_hox = (sink_h + (sink_oh+sink_ho2)*self.o_to_o3_ratio) \
            * self.nm* 3600*24 # ppv/day
        


        ####################### Nitrogen Partitioning #########################
        #
        # First work out how much NOx there is from NOy
        # HNO3 sources and sinks (part of NOy but faster than above)
        #  k_no2_oh:    NO2 + OH + M -> HNO3 + M  Source (reservoir of NO2)
        #  jhno3: HNO3 + hv -> NO2 + OH  Sink
        #  k8:    HNO3 + OH + M -> H2O + NO3 + M Sink
        #
        # Note that k7 and jhno3 modify HOx and k8 produced water vapour
        #
        # Next partition the NOx into NO and NO2
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

        b3 = 5.1E-12*np.exp(   210 / self.lyT )/100**3         # (m^3/molec)/s
        b4 = 3.0E-12*np.exp( -1500 / self.lyT )/100**3         # (m^3/molec)/s

        #no_to_no2_ratio = b3*self.o_to_o3_ratio/b4           # unitless
        odd_o =  self.o_to_o3_ratio * self.lyO * self.nm           #density of O(1D)
        no_to_no2_ratio = (b3*odd_o + self.jno2)/(b4*self.lyO*self.nm)           # unitless
        no_to_nox_ratio = no_to_no2_ratio / (1 + no_to_no2_ratio)
        no2_to_nox_ratio = 1-no_to_nox_ratio 

        lyNOy = self.D['lyNOy']
        lyN2O5 = self.D['lyN2O5']
        lyClONO2 = self.D['lyClONO2']

        #  k_no2_oh:    NO2 + OH + M -> HNO3 + M  Source (reservoir of NO2)
        # k_a to HONO2
        k_a_0 = 1.8e-30 * ((298/self.lyT)**(-3))/(100**6)       # (m^3/molec)^2/s
        k_a_inf = 2.8e-11 /(100**3)                             # m^3/molec/s
        k_a = self.termolecular_k_calc(k_a_0, k_a_inf,self.nm)                            # (m^3/molec)/s
        # k_b to HOONO
        k_b_0 = 9.3e-32 * ((298/self.lyT)**(-3.9))/(100**6)     # (m^3/molec)^2/s
        k_b_inf = 4.2e-11 * ((298/self.lyT)**(-0.5))/(100**3)   # m^3/molec/s
        k_b = self.termolecular_k_calc(k_b_0, k_b_inf, self.nm)                            # (m^3/molec)/s
        self.k_no2_oh = k_a + k_b                                          # (m^3/molec)/s
        
        #  k_hno3_oh:    HNO3 + OH + M -> H2O + NO3 + M Sink
        k_0 = 3.9e-31 * ((298/self.lyT)**(7.2))/(100**6)         # (m^3/molec)^2/s
        k_inf = 1.5e-13 * ((298/self.lyT)**(4.8))/(100**3)      # m^3/molec/s
        k_int = 3.7e-14 * np.exp(240/self.lyT)/(100**3)
        k_f = self.termolecular_k_calc(k_0, k_inf, self.nm)
        k_ca = k_int * (1-k_f/k_inf)
        self.k_hno3_oh = k_f + k_ca                                         # (m^3/molec)/s

        hno3_to_nox_ratio = (self.k_no2_oh*self.lyOH*self.nm \
            / (self.k_hno3_oh*self.lyOH*self.nm + self.jhno3)) * no2_to_nox_ratio
      
        n2o5_to_nox_ratio = lyN2O5 / self.D['lyNOx']
        clono2_to_nox_ratio = lyClONO2 / self.D['lyNOx']
               
        nox_to_noy_ratio = 1/(1 + hno3_to_nox_ratio \
            + 2*n2o5_to_nox_ratio + clono2_to_nox_ratio)

        self.lyNOx = nox_to_noy_ratio * self.D['lyNOy']
        self.lyNO  = no_to_nox_ratio * self.lyNOx
        self.lyNO2  = no2_to_nox_ratio * self.lyNOx
        self.lyHNO3 = hno3_to_nox_ratio * self.lyNOx
       

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
        no_to_no2_ratio = (b3*odd_o + self.jno2)/(b4*self.lyO*self.nm)           # unitless

        sink_nox = (b4*self.lyNO*self.nm \
            + b3*self.lyNO2*self.o_to_o3_ratio*self.nm \
            - self.jno2*self.lyNO2/self.lyO)*86400 # ppv/day

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
        
        # clox = cl + clo
        # cloy = clox + clono2
        self.lyClOx = self.D['lyClOy'] - self.D['lyClONO2']
        self.lyClOx[self.lyClOx<0] = 0.0

        cl1 = 2.8E-11*np.exp(   85 / self.lyT )/100**3         # (m^3/molec)/s
        cl2 = 2.3E-11*np.exp( -200 / self.lyT )/100**3         # (m^3/molec)/s

        cl_to_clo_ratio = cl1*self.o_to_o3_ratio/cl2           # unitless

        if len(np.where(self.o_to_o3_ratio==0)[0])>0:
          cl_to_clox_ratio = 0 
        else:
          cl_to_clox_ratio = ( 1 + 1/cl_to_clo_ratio)**-1  # unitless

        lyCl  = cl_to_clox_ratio * self.lyClOx
        lyClO = (1-cl_to_clox_ratio) * self.lyClOx
        self.lyClO = lyClO
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
        #y

        sink_clox = (cl2*lyCl + cl1*self.lyClO*self.o_to_o3_ratio)*self.nm*86400 # ppv/day

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

        self.sink_hox = sink_hox*self.lyO
        self.source = source       #ppv / day
        self.sink_ox = sink_ox*self.lyO**2
        self.sink_nox = sink_nox*self.lyO
        self.sink_clox = sink_clox*self.lyO
   
    def nighttime_chemistry( self ):
        if self.next_night == 1:  
        # At sunset, odd O disappears and all NOx gets quickly converted to NO2
        # all Cl also goes to ClO
        # Do this once
          self.next_night = 0
          self.D['lyNO2'] = self.D['lyNOx']
          self.D['lyNO'] = self.D['lyNOx']*0.0
          self.D['lyClO'] = self.D['lyClOx']
        
        lyO = self.D['lyO']
        lyNO2 = self.D['lyNO2']
        lyClO = self.D['lyClO']
        
        # NO2 then slowly gets converted to the N2O5 reservoir as
        # k8 :      NO2 + O3 -> NO3 + O2 slow
        # k9:  NO3 + NO2 + M -> N2O5 + M fast
        #
        # The next reaction is
        # 2 NO2 + O3 + M -> N2O5 + O2 + M
        # d[NO2]/dt = -k8 [NO2][O3]
        
        # molecule number density
        nm = ( 6.022E+23 * self.D['lyP'] )                  \
                / ( self.D['molar_mass_air'] * self.D['lyT'] * 287.058 ) 
                # molec/m^3

        lyT = self.D['lyT']
        k8 = 1.2e-13 * np.exp(-2450/self.lyT)/(100**3) # (m^3/molec)/s
     
        self.dlyNO2 = -k8*lyO*nm*lyNO2*self.D['timestepsize']*24*3600
        #2 NO2 produces one N2O5
        self.dlyN2O5 = -0.5*self.dlyNO2

        # NO2 also gets slowly converted to ClONO2 reservoir
        # k10: ClO + NO2 + M -> ClONO2 + M        
        k_0 = 1.8e-31 * ((298/self.lyT)**(-3.4))/(100**6)       # (m^3/molec)^2/s
        k_inf = 1.5e-11 * ((298/self.lyT)**(-1.9))/(100**3)     # m^3/molec/s
        k10 = self.termolecular_k_calc(k_0, k_inf, nm)             # (m^3/molec)/s
        self.dlyClO = -k10*lyClO*nm*lyNO2*self.D['timestepsize']*24*3600
        self.dlyClONO2 = -self.dlyClO

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

        self.dx = ( self.D['lyZ'][1:  ] - self.D['lyZ'][ :-1] )*1000
        
        self.b00 = +1/self.dx
        self.bm1 = -1/self.dx

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
        
        #set chem tropospheric condition to be cti - 12
        self.chem_trop = self.D['cti'] -13

    def compute_dz(self):
        lyZ = self.D['lyZ']
        lvZ = np.zeros(len(lyZ) + 1)

        # interior interfaces
        lvZ[1:-1] = 0.5 * (lyZ[:-1] + lyZ[1:])

        # boundaries (linear extrapolation)
        lvZ[0]  = lyZ[0]  - 0.5 * (lyZ[1] - lyZ[0])
        lvZ[-1] = lyZ[-1] + 0.5 * (lyZ[-1] - lyZ[-2])

        # now compute dz
        dz = lvZ[1:] - lvZ[:-1]
        self.dz = dz*1000
        self.lvZ = lvZ

    def advect_upwind(self, C, w):
        """
        1D Explicit Upwind Advection for chemical species.
        Positivity-preserving (no wobbles), but strictly bound by the CFL condition.
        
        Parameters:
        C  : 1D numpy array of species concentrations at current timestep
        w  : 1D numpy array of vertical velocities (positive = upward) in m/s
        dz : Grid spacing in meters (can be a scalar or array)
        
        Returns:
        C_new : 1D numpy array of concentrations at the next timestep
        """
        # Calculate the Courant number
        c = w * (self.D['timestepsize'] / self.dz)
        
        max_courant = np.max(np.abs(c))
        if max_courant > 1.0:
            raise ValueError(
                f"CFL Condition violated! Max Courant = {max_courant:.2f}. "
                "The explicit scheme will blow up. You must reduce your timestep (dt) "
                "or increase your grid spacing (dz)."
            )
        
        # Separate into upward (positive) and downward (negative) wind components
        c_pos = np.maximum(c, 0)
        c_neg = np.minimum(c, 0) 
        
        # Initialize array for the change in concentration
        dC = np.zeros_like(C)
        
        # 1. Advect Upward (w > 0)
        # Cell i gains from i-1, and loses its own concentration to i+1
        dC[1:] += c_pos[1:] * C[:-1] - c_pos[1:] * C[1:]
        dC[0]  -= c_pos[0] * C[0]  # Bottom boundary loss
        
        # 2. Advect Downward (w < 0)
        # Cell i gains from i+1, and loses its own concentration to i-1
        # (Note: c_neg is a negative number, so we subtract it to add concentration)
        dC[:-1] -= c_neg[:-1] * C[1:] - c_neg[:-1] * C[:-1]
        dC[-1]  += c_neg[-1] * C[-1] # Top boundary loss
        
        return dC


    def ozone_transport( self ):

        #dodz = np.diff( self.D['lyO'] ) / np.diff( self.D['lyZ']*1000 )
        #dodz = ( dodz[1:] + dodz[:-1] ) / 2
        #dodz = np.insert( dodz , 0 , 0 )
        #dodz = np.append( dodz , 0 )

        #dfdx =    self.bm1 * self.D['lyO'][ :-1] \
        #        + self.b00 * self.D['lyO'][1:  ] \

        #dfdx = np.insert( dfdx , 0 , 0 )

        ##
        ## This finite difference may be causing the wobbles in the ozone profile
        ## and you can perhaps replace it with a difference that uses the i+1/2
        ## and i - 1/2 points around each point i. These are the level values,
        ## and they should probably be related to the average values in some
        ## exponential terms.
        ##

        #lydn2o = self.advect_upwind(self.D['lyN2O'], self.lyU_lyO )
        #self.lydo = -dfdx*self.lyU_lyO
        
        lydo = self.advect_upwind(self.D['lyO'], self.lyU_lyO )
        self.D['lyO'] += self.D['timestepsize'] * lydo
    
    def n2o_transport( self ):
        self.D['lyN2O'][:self.chem_trop]= 0.315e-6 
        #dfdx =    self.bm1 * self.D['lyN2O'][:-1] \
        #        + self.b00 * self.D['lyN2O'][1:] \

        #dfdx = np.insert( dfdx , 0 , 0 )
        #lydn2o = -dfdx*self.lyU_lyN2O
        lydn2o = self.advect_upwind(self.D['lyN2O'], self.lyU_lyO )

        self.D['lyN2O'][self.chem_trop:] += self.D['timestepsize'] * lydn2o[self.chem_trop:]
    
    def noy_transport( self ):
        lydnoy = self.advect_upwind(self.D['lyNOy'], self.lyU_lyO )
        self.D['lyNOy'] += self.D['timestepsize'] * lydnoy
    

    def n2o_minimum( self ):
        self.D['lyN2O'][self.D['lyN2O']<0] = 20.0e-12
    
    def noy_minimum( self ):
        self.D['lyNOy'][self.D['lyNOy']<0] = 0.3e-9
    
    def n2o5_minimum( self ):
        self.D['lyN2O5'][self.D['lyN2O5']<0] = 0.0
    
    def clono2_minimum( self ):
        self.D['lyClONO2'][self.D['lyClONO2']<0] = 0.0
    
    def clo_minimum( self ):
        self.D['lyClO'][self.D['lyClO']<0] = 0.0
    
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

        if self.D['same_day']<0: 
            time = self.D['day'] % 365 
        else:
            time = ( self.D['day'] % 1 ) + self.D['same_day']

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
        if self.D['same_day']<0: 
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
        self.lyU_lyN2O = self.D['lyZ']*0.0 + self.D['strat_up_ozone'] / 1000 * 86400
        #self.lyU_lyHNO3 = self.D['lyZ']*0.0 + self.D['strat_up_ozone'] / 1000 * 86400
        self.lyU_lyNOy = self.D['lyZ']*0.0 + self.D['strat_up_ozone'] / 1000 * 86400
        
        if self.bool_impose_resw:
          
          current_qbo_time = self.computed_days % (28*30)
          i_currenttime = int(current_qbo_time)
          dtime = current_qbo_time - i_currenttime
          #resw units of m/day
          resw  = (dtime*self.resw_in(i_time=i_currenttime+1)[:] \
              + (1-dtime)*self.resw_in(i_time=i_currenttime)[:])\
              / 1000 * 86400 

          self.lyU_lyT += resw.squeeze()  #resw S term
          self.lyU_lyO += resw.squeeze()  #resw dO3/dz term
          self.lyU_lyN2O += resw.squeeze()  #resw dO3/dz term
          #self.lyU_lyHNO3 += resw.squeeze()  #resw dO3/dz term
          self.lyU_lyNOy += resw.squeeze()  #resw dO3/dz term


        if self.bool_temps_convection: 
          self.lyU_lyT[:self.D['cti']] = 0.0

    def update_experimental_variables( self ):

        for var in self.exp_strs.keys():
            exec("self.D['"+var+"']="+self.exp_strs[var])
        
    def assign_experimental_command( self, var, string ):

        self.exp_strs[var] = string

    def timestep( self ):
        self.count+=1
        # Set independent variables with varying values
        self.update_experimental_variables()

        # Check if we are in PCE mode only
        if self.bool_pce_only:
            self.set_pce_only_flags()

        # Time adjustments
        self.calculate_sza()
        self.set_upwelling()
        self.set_mmair()
        
        # Check if we are in fdh mode
        if self.bool_fdh:
            self.set_fdh_flags()
        else:
            self.set_water()

        # Molecular adjustments aside from ozone
        #self.set_pressure()
        #self.set_broad()
        #self.set_water()


        # Radiation
        if self.bool_run_rad:
            ##run rad once with no T changes
            #self.lyT = self.D['lyT_0']
            #self.lvT = self.D['lvT_0']
            #self.lyO = self.D['lyO']

            #self.set_pressure()
            #self.set_broad()
            #self.run_rad()
            #self.D['lyHS_T0']=self.lyHS
            #self.D['lyHL_T0']=self.lyHL

            ##run rad again with no O3 changes
            #self.lyT = self.D['lyT']
            #self.lvT = self.D['lvT']
            #self.lyO = self.D['lyO_0']

            #self.set_pressure()
            #self.set_broad()
            #self.run_rad()
            #self.D['lyHS_O0']=self.lyHS
            #self.D['lyHL_O0']=self.lyHL

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
                ##run chemistry with T constant
                #self.lyT = self.D['lyT_0']
                #self.lyO = self.D['lyO']
                #self.reaction_rates()
                #self.noy_chemistry()
                #self.ozone_chemistry()
                #self.D['sink_hox_T0'] = self.sink_hox
                #self.D['source_T0'] = self.source       #ppv / day
                #self.D['sink_ox_T0'] = self.sink_ox
                #self.D['sink_nox_T0'] = self.sink_nox
                #self.D['sink_clox_T0'] = self.sink_clox

                ##run chemistry with O3 constant
                #self.lyT = self.D['lyT']
                #self.lyO = self.D['lyO_0']
                #self.ozone_chemistry()
                #self.D['sink_hox_O0'] = self.sink_hox
                #self.D['source_O0'] = self.source       #ppv / day
                #self.D['sink_ox_O0'] = self.sink_ox
                #self.D['sink_nox_O0'] = self.sink_nox
                #self.D['sink_clox_O0'] = self.sink_clox

                #run chemistry normally
                self.lyT = self.D['lyT']
                self.lyO = self.D['lyO']
                self.reaction_rates()
                self.noy_chemistry()
                self.D['lyN2O'] += self.dlyN2O
                self.D['jn2o'] = self.jn2o
                self.D['jn2o5'] = self.jn2o5
                self.D['lyNOx'] += self.dlyNOx
                self.D['lyNOy'] += self.dlyNOy
                self.D['lyN2O5'] += self.dlyN2O5
                self.D['lyClONO2'] += self.dlyClONO2
                self.D['sink_n2o'] = self.sink_n2o

                self.ozone_chemistry()
                self.D['lyO'] += self.dlyO
                self.D['jhno3'] = self.jhno3
                self.D['lyHNO3'] = self.lyHNO3
                self.D['lyOH'] = self.lyOH
                self.D['nm'] = self.nm
                self.D['j3'] = self.j3
                self.D['j2_hz'] = self.j2_hz
                self.D['j2_srb'] = self.j2_srb
                self.D['j2'] = self.j2
                self.D['lyNO'] = self.lyNO
                self.D['lyNO2'] = self.lyNO2
                self.D['lyNOx'] = self.lyNOx
                self.D['lyClO'] = self.lyClO
                self.D['lyClOx'] = self.lyClOx
                self.D['sink_hox'] = self.sink_hox
                self.D['source'] = self.source       #ppv / day
                self.D['sink_ox'] = self.sink_ox
                self.D['sink_nox'] = self.sink_nox
                self.D['sink_clox'] = self.sink_clox
                
                self.next_night = 1
            
            else:
                self.D['source']    = self.D['source']*0.0
                self.D['sink_ox']   = self.D['sink_ox']*0.0
                self.D['sink_nox']  = self.D['sink_nox']*0.0
                self.D['sink_hox']   = self.D['sink_hox']*0.0
                self.D['lyOH'] = self.D['lyHOx']*0.0
                
                self.D['sink_n2o']   = self.D['sink_n2o']*0.0
                
                self.nighttime_chemistry()
                # NOy = NOx + HNO3 + 2 N2O5 + ClONO2 
                # NOy doesn't change during the night. 2NO2 just goes to N2O5.
                # NO2 also goes to ClONO2
                self.D['lyNO2'] += self.dlyNO2
                self.D['lyNOx'] = self.D['lyNO2']
                self.D['lyN2O5'] += self.dlyN2O5 

                self.D['lyClO'] += self.dlyClO 
                self.D['lyClONO2'] += self.dlyClONO2 

            self.n2o_transport()
            self.noy_transport()

            self.ozone_transport()
            self.ozone_minimum()
            self.n2o_minimum()
            self.noy_minimum()
            self.n2o5_minimum()
            self.clono2_minimum()
            self.clo_minimum()

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
        self.hno3_chemistry_initialize()
        self.n2o_chemistry_initialize()
        self.no2_chemistry_initialize()
        self.n2o5_chemistry_initialize()
        self.clono2_chemistry_initialize()
        
        self.set_transport_weights()
        self.compute_dz()

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

