import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

chars = [ [ 1, 0, 1],
          [0, 1, 0],
          [1, 0, 1] ]

norm_chars = normalize( chars )

print norm_chars


 
