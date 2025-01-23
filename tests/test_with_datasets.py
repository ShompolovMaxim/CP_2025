import sys

sys.path.append('../')

from numpy import genfromtxt
import numpy as np
from SuppressionKAnonimityTimeOptimal import SuppressionKAnonymityTimeOptimal


df = genfromtxt('../static/Bank_Personal_Loan_Modelling.csv', delimiter=',')
df = np.delete(df, (0), axis=0)
df = np.delete(df, (0, 8, 9, 10, 11, 12, 13), axis=1)
sdf, n_sup = SuppressionKAnonymityTimeOptimal(2).depersonalize(df)
print(sdf[0:5])
print(n_sup)

