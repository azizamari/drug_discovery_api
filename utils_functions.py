import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem

def get_model():
    model = pickle.load(open('model.sav', 'rb'))
    return model

def clean_smiles(i):
  cpd = str(i).split('.')
  cpd_longest = max(cpd, key = len)
  return cpd_longest

def lipinski(smiles):
    moldata= Chem.MolFromSmiles(smiles) 
    
    desc_MolWt = Descriptors.MolWt(moldata)
    desc_MolLogP = Descriptors.MolLogP(moldata)
    desc_NumHDonors = Lipinski.NumHDonors(moldata)
    desc_NumHAcceptors = Lipinski.NumHAcceptors(moldata)
        
    descriptors = {'MW':desc_MolWt,'LogP':desc_MolLogP,'NumHDonors':desc_NumHDonors, 'NumHAcceptors':desc_NumHAcceptors}
    
    return descriptors

def calculate_rotatable_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # add hydrogens
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # generate 3D coordinates
    AllChem.UFFOptimizeMolecule(mol)  # optimize 3D structure
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    return num_rotatable_bonds

def get_rotatable_bonds(smiles):
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
      num_rotatable_bonds = None
  else:
      mol = Chem.AddHs(mol)  # add hydrogens
      try:
          AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # generate 3D coordinates
          AllChem.UFFOptimizeMolecule(mol)  # optimize 3D structure
          num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
      except ValueError:
          num_rotatable_bonds = None
  return {'rotatable_bonds':num_rotatable_bonds}

def calculate_num_rings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    num_rings = Descriptors.RingCount(mol)
    return {'num_rings':num_rings}

def calculate_polarizability(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {"polarizabilities":Descriptors.MolMR(mol)}

def calculate_num_chiral_centers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    chiral_centers = Chem.FindMolChiralCenters(mol)
    num_chiral_centers = len(chiral_centers)
    return {"num_chiral_centers_list":num_chiral_centers}

def calculate_num_heavy_atoms(smile):
    mol = Chem.MolFromSmiles(smile)
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    return {"NumHeavyAtoms":num_heavy_atoms}

def input_pipeline(smiles):
  smiles=clean_smiles(smiles)
  features={'canonical_smiles':smiles}
  features|=calculate_num_heavy_atoms(smiles)
  features|=calculate_num_chiral_centers(smiles)
  features|=calculate_num_rings(smiles)
  features|=get_rotatable_bonds(smiles)
  features|=lipinski(smiles)
  return features


smiles_example="CC(C)C[C@H](NC(=O)[C@@H](NC(=O)[C@@H](N)CCC(=O)O)C(C)C)C(=O)N[C@@H](Cc1ccccc1)[C@@H](O)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](C)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O"
s=input_pipeline(smiles_example)
print(s)