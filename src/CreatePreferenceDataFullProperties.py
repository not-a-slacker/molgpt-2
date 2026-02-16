# %%
from dockstring import load_target
import multiprocessing as mp


def dock_smile(smile):
    target = load_target('LCK')
    try:
        score, _ =  target.dock(smile)
    except:
        score = None
    return score

# with mp.Pool(mp.cpu_count()) as p:
#     results = p.map(dock_smile, smilelist)

# %%
config = {
    'batch_size' : 128,
    'd_model': 512,
    'n_heads': 8,
    'n_layers':6,
    'hidden_units': 512,
    'lr': 1e-5,
    'epochs': 265,
    'properties': sorted(['affinity','logps', 'qeds', 'sas', 'tpsas']),
}
config['run_name'] = "LCK_DOCKSTRING_FAST_ACTUAL_"+ "_".join(prop for prop in config['properties'])
print(config)

# %%
import pickle
with open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities_combined.pkl', 'rb') as f:
    data = pickle.load(f)

# %%
print(len(data))

# %%

import rdkit
import rdkit.Chem as Chem

from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
from rdkit.Chem import QED, Descriptors, Crippen

def compute_logp(mol):
    try:
        logp = Crippen.MolLogP(mol)
    except:
        logp = None
    return logp

def compute_qed(mol):
    try:
        qed = QED.qed(mol)
    except:
        qed = None
    return qed

def compute_tpsa(mol):
    try:
        tpsa = Descriptors.TPSA(mol)
    except:
        tpsa = None
    return tpsa

def compute_sas(mol):
    try:
        sas = sascorer.calculateScore(mol)
    except:
        sas = None
    return sas

def compute_props(smile):
    # try:
    #     mol = Chem.MolFromSmiles(smile)
    # except:
    #     return [None, None, None]
    mol = Chem.MolFromSmiles(smile)
    logp = compute_logp(mol)
    qed = compute_qed(mol)
    tpsa = compute_tpsa(mol)
    sas = compute_sas(mol)
    return [logp, qed, tpsa, sas]


# %%
print(len(data))

# %%
print(data[100])



# %%
import time
start_time = time.time()
all_data = []
for i, row in enumerate(data):
    print(" iter ",i)
    target_smile = row[0]
    target_properties = row[1]
    candidate_smiles = row[2]
    candidate_affinities = row[3]
    with mp.Pool(mp.cpu_count()) as p:
        results = p.map(compute_props, candidate_smiles)
    all_data.append([target_smile,
                    target_properties,
                    candidate_smiles,
                    candidate_affinities,
                    results])
    if len(all_data) % 100== 0:
        print(" Completed ", len(all_data), " entries")
        print(" Time elapsed: ", time.time() - start_time)
        with open('../checkpoints/'+config['run_name']+'/PreferenceDataFullProperties.pkl', 'wb') as f:
            pickle.dump(all_data, f)

# %%

import pickle
with open('../checkpoints/'+config['run_name'] +'/PreferenceDataFullProperties.pkl', 'rb') as f:
    all_data = pickle.load(f)

# %%
print(f" Total entries in all_data: {len(all_data)}")

# %%
print(f" Sample entry in all_data: {all_data[0]}")

# %%
import pandas as pd
df = pd.read_csv('../data/lck_dockstring_data1.csv')

# %%
import sklearn
# minmaxscaler
from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df['qeds'] = scaler.fit_transform(df['qeds'].values.reshape(-1,1))
# df['tpsas'] = scaler.fit_transform(df['tpsas'].values.reshape(-1,1))
# df['logps'] = scaler.fit_transform(df['logps'].values.reshape(-1,1))
# df['affinity'] = scaler.fit_transform(df['affinity'].values.reshape(-1,1))

affinity_scaler = MinMaxScaler()
qed_scaler = MinMaxScaler()
logp_scaler = MinMaxScaler()
tpsas_scaler = MinMaxScaler()
sas_scaler = MinMaxScaler()

affinity_scaler.fit(df['affinity'].values.reshape(-1,1))
qed_scaler.fit(df['qed'].values.reshape(-1,1))
logp_scaler.fit(df['logp'].values.reshape(-1,1))
tpsas_scaler.fit(df['tpsa'].values.reshape(-1,1))
sas_scaler.fit(df['sas'].values.reshape(-1,1))

df['qed'] = qed_scaler.transform(df['qed'].values.reshape(-1,1))
df['logp'] = logp_scaler.transform(df['logp'].values.reshape(-1,1))
df['tpsa'] = tpsas_scaler.transform(df['tpsa'].values.reshape(-1,1))
df['affinity'] = affinity_scaler.transform(df['affinity'].values.reshape(-1,1))
df['sas'] = sas_scaler.transform(df['sas'].values.reshape(-1,1))

# %%

print(all_data[1][0])  # target smile
print(all_data[1][1])  # target properties
print(all_data[1][2])  # candidate smiles
print(all_data[1][3])  # candidate affinities
print(all_data[1][4])  # candidate properties

# %%
def compute_preference_score(target_properties, candidate_properties):
    # compute the preference score
    # target_properties = [aff, logp, qed, tpsa]
    # candidate_properties = [aff, logp, qed, tpsa]
    # preference_score = 1 - (1/4) * (abs(aff - aff') + abs(logp - logp') + abs(qed - qed') + abs(tpsa - tpsa'))
    # return 1 - (1/4) * np.sum(np.abs(target_properties - candidate_properties))
    return 1 - (1/5) * np.sum(np.abs(target_properties - candidate_properties))
    # print(np.abs(target_properties[0] - candidate_properties[0]), 1 - np.abs(target_properties[0] - candidate_properties[0]))
    # return 1 - np.abs(target_properties[0] - candidate_properties[0])
    # return 1 - (1/2) * (np.abs(target_properties[0] - candidate_properties[0]) + np.abs(target_properties[1] - candidate_properties[1]))

import numpy as np
PreferenceData = []
for i, row in enumerate(all_data):
    target_smile = row[0]
    target_properties = np.array(row[1])
    candidate_smiles = np.array(row[2])
    candidate_affinities = np.array(row[3])
    candidate_properties = np.array(row[4])
    
    tuples = []
    for smi, aff, prop in zip(candidate_smiles, candidate_affinities, candidate_properties):
        tuples.append([aff] + list(prop)) 
    tuples = np.array(tuples)
    
    scaled_affs = affinity_scaler.transform(tuples[:,0].reshape(-1,1))
    scaled_logps = logp_scaler.transform(tuples[:,1].reshape(-1,1))
    scaled_qeds = qed_scaler.transform(tuples[:,2].reshape(-1,1))
    scaled_tpsas = tpsas_scaler.transform(tuples[:,3].reshape(-1,1))
    scaled_sas = sas_scaler.transform(tuples[:,4].reshape(-1,1))
   
    tuples = np.concatenate([scaled_affs, scaled_logps, scaled_qeds, scaled_sas, scaled_tpsas], axis=1)

    
    preference_scores = []
    for j in range(len(tuples)):
        score = compute_preference_score(target_properties, tuples[j])
        if not np.isnan(score):
            # print(score)
            preference_scores.append([ candidate_smiles[j], score, tuples[j]])
        else:
            print(" NaN score for candidate ", candidate_smiles[j])
    
    if len(preference_scores) == 0:
        print(" No valid preference scores for entry ", i)
        continue
    preference_scores = sorted(preference_scores, key=lambda x: x[1], reverse=True)
    print(f"len (preference_scores) for entry {i}: ", len(preference_scores))
    good_sample = preference_scores[0]
    bad_sample = preference_scores[-1]
    PreferenceData.append([target_smile, target_properties, good_sample, bad_sample])
    
    
with open('../checkpoints/'+config['run_name']+'/PreferenceData.pkl', 'wb') as f:
    pickle.dump(PreferenceData, f)

# %%
print(f" Total entries in PreferenceData: {len(PreferenceData)}")

# %%
print(f" Sample entry in PreferenceData: {PreferenceData[0]}")
