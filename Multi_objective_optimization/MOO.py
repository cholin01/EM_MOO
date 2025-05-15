import pandas as pd
import numpy as np
from scipy.stats import norm
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

da = pd.read_csv('./generate_molecules.csv')
dat = da['smiles']
print(len(dat))

def filter_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if Chem.GetFormalCharge(mol) != 0:
        return False
    ring_sizes = set(len(ring) for ring in Chem.GetSymmSSSR(mol))

    if any(ring_size in [3, 4] for ring_size in ring_sizes):
        return False
    return True

filtered_dat = []
filtered_idx = []
for i, smiles in enumerate(dat):
    if filter_molecule(smiles):
        filtered_dat.append(smiles)
        filtered_idx.append(i)
print(len(filtered_dat))

Y1 = np.load("./pred_Q.npy")
Y2 = np.load("./pred_bde.npy")
Y2res = Y2[:,0]
Y2res
Y1res = Y1[:,0]
Y1res
inputPoints = [list(item) for item in zip(Y1res, Y2res)]

s1 = Y1[:, 1]
means1 = Y1[:, 0]

s2 = Y2[:, 1]
means2 = Y2[:, 0]

PI1 = norm.cdf((1896 - means1) / s1) + norm.cdf((120 - means2) / s2) - (norm.cdf((1800 - means1) / s1)) * (
    norm.cdf((120 - means2) / s2))

PI2 = norm.cdf((1896 - means1) / s1) + norm.cdf((450 - means2) / s2) - (norm.cdf((1800 - means1) / s1)) * (
    norm.cdf((400 - means2) / s2))
PI = PI2 - PI1


idx1 = np.argsort(PI)[::-1][:2000]

print(" Top ：", PI[idx1])
print("index：", idx1)
