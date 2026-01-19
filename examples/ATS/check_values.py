
import sys
import os
import numpy as np
import pprint as pp

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from ats_device import get_ats_parameter, working_spot

dc, ac_plus, ac_minus, omega = get_ats_parameter(*working_spot)

print("AC Plus:")
pp.pprint(ac_plus)
print("\nAC Minus:")
pp.pprint(ac_minus)
