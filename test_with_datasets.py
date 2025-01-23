import numpy as np
from numpy import genfromtxt
from suppression_with_k_anonimity_baseline import *


df = genfromtxt('Bank_Personal_Loan_Modelling.csv', delimiter=',')
df = np.delete(df, (0), axis=0)
df = np.delete(df, (0, 8, 9, 10, 11, 12, 13), axis=1)
sdf, n_sup = suppression_k_anonimity_time_optimal(df, 2)
print(sdf[0:5])
print(n_sup)

