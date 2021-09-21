from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import *
from rdkit.Chem import *
from rdkit import RDLogger    
from rdkit.Chem.Crippen import MolLogP,MolMR
import seaborn as sns
import pickle as pkl
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import collections
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')         
sns.set_style('ticks')
sns.set_style('white')


# Flatten nested lists 
def flatten(x):
    if isinstance(x,collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

    
# Return the names of the descriptors given a list of indices    
def index_to_name(descriptors,idxs):
    return np.array(descriptors)[idxs]


# Get train/test split indices for dataframe
def idx_list(df_idx,split_idx):
    return list(np.array(df_idx)[split_idx])  
    

# Convert a single SMILES to RDKIT molecule format
def SMILES2MOL(mole):
    return Chem.MolFromSmiles(mole)


# Convert an array/vector of SMILES data to RDKIT mole format
def SMILES2MOLES(moles):
    vectSMILES2MOL=np.vectorize(SMILES2MOL)
    return vectSMILES2MOL(moles)

# Add hydrogens to molecule
def ADDH_MOL(mole):
    return Chem.AddHs(mole)

# Add hydrogens to a list of molecules
def ADDH_MOLES(moles):
    vectADDH_MOL=np.vectorize(ADDH_MOL)
    return vectADDH_MOL(moles)

    
# Get 67 descriptors for a molecule
def STDD(molecules):
    
    std_descriptors=[]
    no_descriptor=[]

    for i,m in enumerate(molecules):
        
        try:
            AllChem.EmbedMolecule(m)
            AllChem.UFFOptimizeMolecule(m,maxIters=1000)
            desc=[]
            desc.append(Chem.GraphDescriptors.BalabanJ(m))        
            desc.append(Chem.GraphDescriptors.BertzCT(m))         
            desc.append(Chem.GraphDescriptors.HallKierAlpha(m))   
            desc.append(Chem.GraphDescriptors.Ipc(m,avg=True))
            desc.append(Chem.Crippen.MolLogP(m))                  
            desc.append(Chem.Crippen.MolMR(m))                    
            desc.append(Chem.Descriptors.ExactMolWt(m))
            desc.append(Chem.Descriptors.FpDensityMorgan1(m))
            desc.append(Chem.Descriptors.FpDensityMorgan2(m))
            desc.append(Chem.Descriptors.FpDensityMorgan3(m))
            desc.append(Chem.Descriptors.HeavyAtomMolWt(m))
            desc.append(Chem.Descriptors.MaxPartialCharge(m))
            desc.append(Chem.Descriptors.MinPartialCharge(m))
            desc.append(Chem.Descriptors.MolWt(m))
            desc.append(Chem.Descriptors.NumValenceElectrons(m))
            desc.append(Chem.Lipinski.HeavyAtomCount(m))
            desc.append(Chem.Lipinski.NumHAcceptors(m))
            desc.append(Chem.Lipinski.NumHDonors(m))
            desc.append(Chem.Lipinski.NHOHCount(m))
            desc.append(Chem.Lipinski.NOCount(m))
            desc.append(Chem.Lipinski.NumAliphaticCarbocycles(m))
            desc.append(Chem.Lipinski.NumAliphaticHeterocycles(m))
            desc.append(Chem.Lipinski.NumAliphaticRings(m))
            desc.append(Chem.Lipinski.NumAromaticCarbocycles(m))
            desc.append(Chem.Lipinski.NumAromaticHeterocycles(m))
            desc.append(Chem.Lipinski.NumAromaticRings(m))
            desc.append(Chem.Lipinski.NumSaturatedCarbocycles(m))
            desc.append(Chem.Lipinski.NumSaturatedHeterocycles(m))
            desc.append(Chem.Lipinski.NumSaturatedRings(m))
            desc.append(Chem.Lipinski.NumHeteroatoms(m))
            desc.append(Chem.Lipinski.NumRotatableBonds(m))
            desc.append(Chem.Lipinski.RingCount(m))
            desc.append(Chem.Lipinski.FractionCSP3(m))
            desc.append(Chem.MolSurf.LabuteASA(m))                
            desc.append(Chem.GraphDescriptors.Chi0(m))
            desc.append(Chem.GraphDescriptors.Chi0n(m))
            desc.append(Chem.GraphDescriptors.Chi0v(m))
            desc.append(Chem.GraphDescriptors.Chi1(m))
            desc.append(Chem.GraphDescriptors.Chi1n(m))
            desc.append(Chem.GraphDescriptors.Chi1v(m))
            desc.append(Chem.GraphDescriptors.Chi2n(m))
            desc.append(Chem.GraphDescriptors.Chi2v(m))
            desc.append(Chem.GraphDescriptors.Chi3n(m))
            desc.append(Chem.GraphDescriptors.Chi3v(m))
            desc.append(Chem.GraphDescriptors.Chi4n(m))
            desc.append(Chem.GraphDescriptors.Chi4v(m))
            desc.append(Chem.rdMolDescriptors.CalcAsphericity(m))
            desc.append(Chem.rdMolDescriptors.CalcEccentricity(m))
            desc.append(Chem.rdMolDescriptors.CalcInertialShapeFactor(m))
            desc.append(Chem.rdMolDescriptors.CalcKappa1(m))
            desc.append(Chem.rdMolDescriptors.CalcKappa2(m))
            desc.append(Chem.rdMolDescriptors.CalcKappa3(m))
            desc.append(Chem.rdMolDescriptors.CalcNPR1(m))
            desc.append(Chem.rdMolDescriptors.CalcNPR2(m))
            desc.append(Chem.rdMolDescriptors.CalcNumAmideBonds(m))
            desc.append(Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(m))
            desc.append(Chem.rdMolDescriptors.CalcNumHeterocycles(m))
            desc.append(Chem.rdMolDescriptors.CalcNumLipinskiHBA(m))
            desc.append(Chem.rdMolDescriptors.CalcNumLipinskiHBD(m))
            desc.append(Chem.rdMolDescriptors.CalcNumSpiroAtoms(m))
            desc.append(Chem.rdMolDescriptors.CalcPBF(m))
            desc.append(Chem.rdMolDescriptors.CalcPMI1(m))
            desc.append(Chem.rdMolDescriptors.CalcPMI2(m))
            desc.append(Chem.rdMolDescriptors.CalcPMI3(m))
            desc.append(Chem.rdMolDescriptors.CalcRadiusOfGyration(m))
            desc.append(Chem.rdMolDescriptors.CalcSpherocityIndex(m))
            desc.append(Chem.rdMolDescriptors.CalcTPSA(m))        
            desc=flatten(desc)
            std_descriptors.append(desc)
            print('molecular descriptors obtained for:',i,flush=True)
            
        except:
            print('molecular descriptors not obtained for:',i,flush=True)
            no_descriptor.append(i)

    return std_descriptors,no_descriptor


# Return a 1d array of canonical smiles for a vector of smiles data
def canonicals(mols):
    def canonical(mol):
        return Chem.MolToSmiles(Chem.MolFromSmiles(mol))
    vector=np.vectorize(canonical)
    return vector(mols)


# Function to get appropriate descriptor
def get_descriptor(dtype,data_SD,train_idx,test_idx):
    if dtype=='PC':
        return pc_descriptor(data_SD,train_idx,test_idx)
    elif dtype=='PCAS':
        return pca_descriptor(data_SD,train_idx,test_idx)
    elif dtype=='PFAS':
        return pfa_descriptor(data_SD,train_idx,test_idx)
    elif dtype=='UD':
        return uncorr_descriptor(data_SD,train_idx,test_idx)
    elif dtype=='67':
        return data_SD.iloc[train_idx+test_idx]

# Get principal components to be used as descriptors
def pc_descriptor(data_SD,train_idx,test_idx,desired_var=0.80):
      
    train_SD=data_SD.iloc[train_idx]
    test_SD=data_SD.iloc[test_idx]
    
    ss=StandardScaler()
    train_scaled=ss.fit_transform(train_SD)
    test_scaled=ss.transform(test_SD)

    pca=PCA(desired_var)
    project_train=pca.fit_transform(train_scaled)
    project_test=pca.transform(test_scaled)
    
    projection=pd.concat([pd.DataFrame(project_train),
                          pd.DataFrame(project_test)])
    projection.index=train_idx+test_idx
    
    return projection

# Get descriptors based on loadings of PCs
def pca_descriptor(data_SD,train_idx,test_idx,desired_var=0.80):
      
    train_SD=data_SD.iloc[train_idx]
    test_SD=data_SD.iloc[test_idx]
    
    ss=StandardScaler()
    train_scaled=ss.fit_transform(train_SD)
    test_scaled=ss.transform(test_SD)

    pca=PCA(desired_var)
    project_train=pca.fit_transform(train_scaled)
    project_test=pca.transform(test_scaled)

    impt_mat=abs(pca.components_)
    feats_idx=[]
    
    for j in range(pca.n_components_):
        cpve=np.cumsum(pca.explained_variance_ratio_)[j]
        if cpve<=0.5:
            feats_idx=feats_idx+list(impt_mat[j,:].argsort()[::-1][:7])
        elif j>=1:
            cpve_m1=np.cumsum(pca.explained_variance_ratio_)[j-1]
            if cpve_m1<0.50 and cpve>0.5:
                feats_idx=feats_idx+list(impt_mat[j,:].argsort()[::-1][:7])
            else:
                feats_idx=feats_idx+list(impt_mat[j,:].argsort()[::-1][:3])
                
    pca_train=train_SD[feats_idx]
    pca_test=test_SD[feats_idx]
    
    return pd.concat([pca_train,pca_test])


# Get descriptors using principal feature analysis
def pfa_descriptor(data_SD,train_idx,test_idx,desired_var=0.90):
    
    train_SD=data_SD.iloc[train_idx]
    test_SD=data_SD.iloc[test_idx]
    
    ss=StandardScaler()
    train_scaled=ss.fit_transform(train_SD)
    test_scaled=ss.transform(test_SD)
    
    cov_mat=np.cov(train_scaled,rowvar=False)
    egnvals,egnvecs=np.linalg.eig(cov_mat)
    total_egnvals=sum(egnvals)
    egnvals_idx=egnvals.argsort()[::-1]   
    egnvals=egnvals[egnvals_idx]
    egnvecs=egnvecs[:,egnvals_idx]
    
    PVE=[(egnv/total_egnvals) for egnv in sorted(egnvals,reverse=True)]
    CPVE=np.cumsum(PVE)
    q=np.where(CPVE>desired_var)[0][0]
    A_q=egnvecs[:,:q].astype(float)
    
    kmeans=KMeans(n_clusters=int(q+3),
                 max_iter=2000,
                 n_init=400,
                 tol=1e-7,
                 random_state=16).fit(A_q)
    feats_idx=[]
    for i,center in enumerate(kmeans.cluster_centers_):
        cluster=np.where(kmeans.labels_==i)[0]
        distances=[]
        for j in cluster:
            distances.append(np.linalg.norm(center-A_q[j,:]))
        feats_idx.append(cluster[np.argsort(distances)[0]])

    pfa_train=train_SD[feats_idx]
    pfa_test=test_SD[feats_idx]
    
    return pd.concat([pfa_train,pfa_test])



# Get descriptors that are minimally correlated based on a PCC cutoff value
def uncorr_descriptor(data_SD,train_idx,test_idx,cutoff=0.8):
    
    data_UD=data_SD.iloc[train_idx]
    corr_np=np.array(abs(data_UD.corr()))
    np.fill_diagonal(corr_np,0)
    removed_idx=[]

    while np.max(corr_np)>cutoff:
        
        r_max=np.max(corr_np)
        feat1=np.where(corr_np==r_max)[0][0]
        feat2=np.where(corr_np==r_max)[1][0]
        r_avg1=np.mean(corr_np[feat1])
        r_avg2=np.mean(corr_np[feat2])

        if r_avg1>r_avg2:
            corr_np[feat1,:]=0
            corr_np[:,feat1]=0
            removed_idx.append(feat1)
        else:
            corr_np[feat2,:]=0
            corr_np[:,feat2]=0
            removed_idx.append(feat2)

    sel_features=[x for x in np.arange(len(data_SD.columns)) if int(x) not in removed_idx]
    uncorr=data_SD[sel_features]
    
    return uncorr.iloc[train_idx+test_idx]


# Function for running the RF regressor
def rf_model(Xtrain,Xtest,Ytrain,Ytest,trees=500,random_state=5):
    
    scaler=StandardScaler()
    Ytrain_scaled=np.ravel(scaler.fit_transform(np.array(Ytrain).reshape(-1,1)))
    rf=RandomForestRegressor(n_estimators=trees,random_state=random_state)
    model=rf.fit(Xtrain,np.ravel(Ytrain_scaled))
    yhat_test=scaler.inverse_transform(model.predict(Xtest))
    
    return [yhat_test,model]


# Function for running decision tree regressor
def dtree_model(Xtrain,Xtest,Ytrain,Ytest,random_state=5):
    
    scaler=StandardScaler()
    Ytrain_scaled=np.ravel(scaler.fit_transform(np.array(Ytrain).reshape(-1,1)))
    rf=DecisionTreeRegressor(random_state=random_state)
    model=rf.fit(Xtrain,Ytrain_scaled)
    yhat_test=scaler.inverse_transform(model.predict(Xtest))
    
    return [yhat_test,model]


# Get the mean/median ensemble prediction
def ensemble_model(yhats):
    return np.mean(yhats,axis=0),np.median(yhats,axis=0)


# Remove nans and infs from dataframe
def remove_nans(mol_data,mol_descriptors):
    
    nan_rows=np.where(np.array(mol_descriptors.isnull())==True)[0]

    mol_data=mol_data.drop(nan_rows)
    mol_data=mol_data.reset_index(drop=True)

    mol_descriptors=mol_descriptors.drop(nan_rows)
    mol_descriptors=mol_descriptors.reset_index(drop=True)

    inf_rows=np.where(np.array(mol_descriptors)==np.inf)[0]

    mol_data=mol_data.drop(inf_rows)
    mol_data=mol_data.reset_index(drop=True)

    mol_descriptors=mol_descriptors.drop(inf_rows)
    mol_descriptors=mol_descriptors.reset_index(drop=True)
    
    return mol_data,mol_descriptors


# Plot setup for the RF results 
def resultsplot_setup(ax,fs,lb,ub,title,axlabel,weight='bold'):
    ax.plot(np.linspace(lb,ub,100),np.linspace(lb,ub,100),color='darkgrey',zorder=0,lw=1)
    ax.plot(np.linspace(lb,ub,100),np.linspace(lb,ub,100)+1,color='lightgrey',zorder=0,lw=1)
    ax.plot(np.linspace(lb,ub,100),np.linspace(lb,ub,100)-1,color='lightgrey',zorder=0,lw=1)
    ax.set_xlim(lb,ub)
    ax.set_ylim(lb,ub)
    ax.set_xlabel('Experimental $log$ $S$',fontsize=fs-1)
    ax.set_ylabel('Predicted $log$ $S$',fontsize=fs-1)
    ax.set_title(title,fontsize=fs)
    ax.text(-10,-10,axlabel,size=fs+1,weight=weight)