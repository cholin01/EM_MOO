from rdkit import Chem
from rdkit.Chem import Descriptors, AddHs
from rdkit.Chem.Descriptors import _descList
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect
from collections import defaultdict
import copy
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from rdkit.Chem.rdmolops import Get3DDistanceMatrix, GetAdjacencyMatrix, GetDistanceMatrix
from rdkit.Chem.Graphs import CharacteristicPolynomial
import matplotlib.pyplot as plt
atom_num_dict = {'C':6,'N':7,'O':8,'H':1,'F':9, 'Cl': 17, 'S': 16 }

#----------------------------------------------------------------------------
def sum_over_bonds_single_mol(mol, bond_types):
    bonds = mol.GetBonds()
    bond_dict = defaultdict(lambda : 0)   # 生成一个字典，对于任意key的查询，value都返回0

    for bond in bonds:
        bond_start_atom = bond.GetBeginAtom().GetSymbol()
        bond_end_atom = bond.GetEndAtom().GetSymbol()
        bond_type = bond.GetSmarts(allBondsExplicit=True)
        if (bond_type == ''):
            bond_type = "-"
        bond_atoms = [bond_start_atom, bond_end_atom]
        bond_string = min(bond_atoms)+bond_type+max(bond_atoms)
        bond_dict[bond_string] += 1

    X_LBoB = [bond_dict[bond_type] for bond_type in bond_types]

    return np.array(X_LBoB).astype('float64')

#----------------------------------------------------------------------------
def literal_bag_of_bonds(mol_list, predefined_bond_types=['N=O', 'N-O', 'C-N', 'C:C', 'C:N', 'N:N', 'C-H', 'H-N', 'N=N', 
                                                          'C-O', 'N#N', 'C-C', 'N-N', 'C=O', 'C#C', 'C#N', 'H-O', 'N:O', 
                                                          'C:O', 'C=N', 'C=C', 'O-O']):
    return sum_over_bonds(mol_list, predefined_bond_types=predefined_bond_types)

#----------------------------------------------------------------------------
def sum_over_bonds(mol_list, predefined_bond_types=['N=O', 'N-O', 'C-N', 'C:C', 'C:N', 'N:N', 'C-H', 'H-N', 'N=N', 
                                                          'C-O', 'N#N', 'C-C', 'N-N', 'C=O', 'C#C', 'C#N', 'H-O', 'N:O', 
                                                          'C:O', 'C=N', 'C=C', 'O-O'], return_names=True):

    if (isinstance(mol_list, list) == False):
        mol_list = [mol_list]

    empty_bond_dict = defaultdict(lambda : 0)
    num_mols = len(mol_list)

    if (len(predefined_bond_types) == 0 ):
        #first pass through to enumerate all bond types in all molecules and set them equal to zero in the dict
        for i, mol in enumerate(mol_list):
            bonds = mol.GetBonds()
            for bond in bonds:
                bond_start_atom = bond.GetBeginAtom().GetSymbol()
                bond_end_atom = bond.GetEndAtom().GetSymbol()
                bond_type = bond.GetSmarts(allBondsExplicit=True)
                bond_atoms = [bond_start_atom, bond_end_atom]
                if (bond_type == ''):
                    bond_type = "-"
                bond_string = min(bond_atoms)+bond_type+max(bond_atoms)
                empty_bond_dict[bond_string] = 0
    else:
        for bond_string in predefined_bond_types:
            empty_bond_dict[bond_string] = 0

    #second pass through to construct X
    bond_types = list(empty_bond_dict.keys())
    num_bond_types = len(bond_types)
    X_LBoB = np.zeros([num_mols, num_bond_types])

    for i, mol in enumerate(mol_list):
        bonds = mol.GetBonds()
        bond_dict = copy.deepcopy(empty_bond_dict)
        for bond in bonds:
            bond_start_atom = bond.GetBeginAtom().GetSymbol()
            bond_end_atom = bond.GetEndAtom().GetSymbol()
            #skip dummy atoms
            if (bond_start_atom=='*' or bond_end_atom=='*'):
                pass
            else:
                bond_type = bond.GetSmarts(allBondsExplicit=True)
                if (bond_type == ''):
                    bond_type = "-"
                bond_atoms = [bond_start_atom, bond_end_atom]
                bond_string = min(bond_atoms)+bond_type+max(bond_atoms)
                bond_dict[bond_string] += 1

        #at the end, pick out only the relevant ones
        X_LBoB[i,:] = [bond_dict[bond_type] for bond_type in bond_types]

    if (return_names):
        return bond_types, X_LBoB
    else:
        return X_LBoB

#----------------------------------------------------------------------------
def truncated_Estate_fingerprints(mol_list):
    X_Estate = []
    feature_index = [15,16,17,18,26,27,28,29,30,31,34,35,36]
    for mol in mol_list:
        x = FingerprintMol(mol)
        x_ =[]
        for index in feature_index:
            x_.append(x[0][index])
        X_Estate.append(x_)
    return X_Estate


#----------------------------------------------------------------------------
def get_num_atom(mol, atomic_number):
    num = 0
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        if (atom_num == atomic_number):
            num += 1
    return num

def oxygen_balance(mol_list):
    x_OB = []
    for mol in mol_list:
        n_C = get_num_atom(mol,6)
        n_H = get_num_atom(mol,1)
        n_O = get_num_atom(mol,8)
        n_N = get_num_atom(mol,7)
        mol_weight = Descriptors.ExactMolWt(mol)
        OB = 1600*(n_O - 2*n_C - 0.5*n_H) / mol_weight
        OB_M = list(np.exp(-OB * 0.01 - np.linspace(0, 5, 20)))
        x_OB.append(OB_M)
    return x_OB

#----------------------------------------------------------------------------
def oxygen_balance_1600(mol):
    '''returns the OB_16000 descriptor'''
    n_O = get_num_atom(mol, 8)
    n_C = get_num_atom(mol, 6)
    n_H = get_num_atom(mol, 1)
    mol_weight = Descriptors.ExactMolWt(mol)
    return round(1600*(n_O - 2*n_C - n_H/2)/mol_weight,2)


def get_neigh_dict(atom):
    '''returns a dictionary with the number of neighbors for a given atom'''
    neighs = defaultdict(int)
    for atom in atom.GetNeighbors():
        neighs[atom.GetSymbol()] += 1
    return neighs


def get_num_with_neighs(mol, central_atom, target_dict):
    '''returns how many atoms of a particular type have a particular configuration of neighbora'''
    target_num = 0
    for key in list(target_dict.keys()):
        target_num += target_dict[key]

    num = 0
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() == central_atom):
            target = True
            nbs = get_neigh_dict(atom)
            for key in list(target_dict.keys()):
                if (nbs[key] != target_dict[key]):
                    target = False
                    break

            n_nbs = len(atom.GetNeighbors())
            if (target_num != n_nbs):
                target = False

            if (target):
                num +=1

    return num


def custom_descriptor_set(mol_list):
    x_cds = []
    for mol in mol_list:
        n_C = get_num_atom(mol, 6)
        n_N = get_num_atom(mol, 7)
        n_O = get_num_atom(mol, 8)
        n_H = get_num_atom(mol, 1)
        n_O1 = get_num_with_neighs(mol, 'O', {'N': 1})
        n_O2 = get_num_with_neighs(mol, 'O', {'N': 1,'C': 1})
        n_O3 = get_num_with_neighs(mol, 'O', {'C': 1})
        #n_O4 = get_num_with_neighs(mol, 'O', {'C': 1,'H': 1})
        n_CNO2 = get_num_with_neighs(mol, 'N', {'O': 2, 'C': 1})
        n_NNO2 = get_num_with_neighs(mol, 'N', {'O': 2, 'N': 1})
        n_ONO = get_num_with_neighs(mol, 'N', {'O': 2})
        n_ONO2 = get_num_with_neighs(mol, 'N', {'O': 3})
        n_CNN = get_num_with_neighs(mol, 'N', {'N': 1, 'C': 1})
        n_CNO = get_num_with_neighs(mol, 'N', {'C': 1,'O': 1})
        n_CNH2 = get_num_with_neighs(mol, 'N', {'C': 1,'H': 2})
        NCratio = round(n_N/n_C,3)

        cds = [oxygen_balance_1600(mol), n_C, n_N, n_O1, n_O2, n_O3, n_H, NCratio, 
                n_CNO2, n_NNO2, n_ONO, n_ONO2, n_CNN, n_CNO, n_CNH2,
              ]
        x_cds.append(cds)
    
    return x_cds

















