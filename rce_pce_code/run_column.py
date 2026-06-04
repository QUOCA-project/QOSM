#import pygeode as pyg
from column import column as column
import numpy as np

file_to_read = 'input.nc'
col = column(file_to_read)

#Perturb fdh

# The model now could run at this point, but you
# probably want to make some small changes now.
col.change_value('totaltime', 200) # 8760 Simulation time to 300 days
col.change_value('recordtime',200)   # Output is mean of last 10 days
#col.change_value('strat_up_temps',0.5)
#col.change_value('strat_up_ozone',0.5)
col.change_value('same_day',1)   # 1:day stays at start day, -1 day evolves
col.change_value('day',152)
col.bool_impose_resw = False
col.bool_impose_nox = False
#col.bool_impose_resw = False


#col.bool_ozone_integration = False #turn off PCE, run in RCE mode
col.bool_ozone_integration = True #turn on PCE
##PCE only mode
#col.bool_pce_only = True

col.change_value('timestepsize',3.0/24)   # Timestep is 1 hour
col.extra['source'][1] = 'ppv/day'
col.extra['sink_ox'][1] = 'ppv/day'
col.extra['sink_nox'][1] = 'ppv/day'
col.wrapper()
col.write_to_file('output.nc')



