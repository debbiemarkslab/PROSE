import pandas as pd 
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

scores_path = 'data/scored_eQTL.csv'
temp = pd.read_csv(scores_path)

def cohen_d(x, y):
    return (np.mean(x) - np.mean(y)) / (np.sqrt((np.var(x) + np.var(y)) / 2))

temp['POET'] = temp['POET'].apply(lambda x: float(x[1:-1]))
# print(temp)
# breakpoint()
a = temp[temp.pip_group == 'causal']['POET'].abs()
b = temp[temp.pip_group == 'background']['POET'].abs()
cohens_d = cohen_d(a, b)
temp['binary_label'] = temp['pip_group'].apply(lambda x: 1 if x == 'causal' else 0)
p,r,_ = precision_recall_curve(temp['binary_label'], temp['POET'].abs()) 
auprc = auc(r, p)

print(f'Cohen\'s d: {cohens_d}')
print(f'AUPRC: {auprc}')