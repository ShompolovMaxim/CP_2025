import sys

sys.path.append('../')

from numpy import genfromtxt
import numpy as np
from SuppressionKAnonymityTimeOptimal import SuppressionKAnonymityTimeOptimal
from Datafly import Datafly
from GeneralizationGreedyByOneEqualSizedGroups import GeneralizationGreedyByOneEqualSizedGroups
from GeneralizationKAnonymityTimeOptimal import GeneralizationKAnonymityTimeOptimal
from GeneralizationKAnonymityGroupJoin import GeneralizationKAnonymityGroupJoin
from SuppressionKAnonymityTimeOptimal import SuppressionKAnonymityTimeOptimal
from utility.metrics import my_by_element_distance
from groupjoin import GroupJoinAggregation, GroupJoinDepersonalizator, GroupJoinTCloseness


df = genfromtxt('../static/Bank_Personal_Loan_Modelling.csv', delimiter=',')
df = np.delete(df, (0), axis=0)
df = np.delete(df, (0, 8, 9, 10, 11, 12, 13), axis=1)
gjmth = GroupJoinAggregation(['real']*6)
gjmtc = GroupJoinTCloseness(2, 1, ['real'])
dep = GroupJoinDepersonalizator(gjmth, gjmtc)
#sdf, n_sup = Datafly(2, ['real']*7, k_suppressed_lines=500).depersonalize(df)
#sdf, n_sup = GeneralizationKAnonymityGroupJoin(2, 4 * ['real'] + ['unordered'] + ['real'] * 2).depersonalize(df)
sdf, n_sup = dep.depersonalize(df, sensitives_ids=[4])
#sdf, n_sup = SuppressionKAnonymityTimeOptimal(k=2).depersonalize(df)
print(sdf[0:5])
print(n_sup)
'''for i in range(len(sdf)):
    if sdf[i][0] is None:
        sdf[i] = df[0]
for i in range(7):
    print(i, len(np.unique(sdf[:,i])), len(np.unique(df[:,i])))'''
print("Distance initial-depersonalized:", my_by_element_distance(df, sdf, 4 * ['real'] + ['real'] + ['real'] * 2))

