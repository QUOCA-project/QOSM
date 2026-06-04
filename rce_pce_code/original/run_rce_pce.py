#import pygeode as pyg
from column_model import column as column
import numpy as np

file_to_read = 'input_file.nc'

col = column(file_to_read)

col.change_value('totaltime' ,4200) # 8760 Simulation time to 300 days
col.change_value('recordtime', 4200)   # Output is mean of last 10 days
col.change_value('eternal_day',1)   # 1:day stays at start day, -1 day evolves
col.change_value('day',152)  # Start on 1 June

col.bool_impose_resw = True # Turn on imposed perturbation to background upwelling

#col.bool_ozone_integration = False #turn off PCE, run in RCE mode
col.bool_ozone_integration = True #turn on PCE
#col.bool_pce_only = True  #Turn off RCE, run in PCE only

col.change_value('timestepsize',1.5/24)   # Timestep is 1.5 hour
col.wrapper()
col.write_to_file('output.nc')



