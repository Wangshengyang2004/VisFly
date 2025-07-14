import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from VisFly.utils.maths import Quaternion



a = Quaternion.from_euler(30/57.3, 0,0)
print(a.R)
print(a.xz_axis)