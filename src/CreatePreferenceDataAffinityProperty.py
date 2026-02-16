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

# %%
config = {
    'batch_size' : 128,
    'd_model': 512,
    'n_heads': 8,
    'n_layers':6,
    'hidden_units': 512,
    'lr': 1e-5,
    'epochs': 265,
    'properties': ['affinity','logps', 'qeds', 'sas', 'tpsas'],
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
print(data[100])

# %%
import pandas as pd
df = pd.read_csv('../data/lck_dockstring_data1.csv')

# %%
import sklearn
# minmaxscaler
from sklearn.preprocessing import MinMaxScaler

affinity_scaler = MinMaxScaler()
affinity_scaler.fit(df['affinity'].values.reshape(-1,1))
df['affinity'] = affinity_scaler.transform(df['affinity'].values.reshape(-1,1))

# %%
import pickle
with open('../checkpoints/'+config['run_name'] +'/PreferenceDataFullProperties.pkl', 'rb') as f:
    all_data = pickle.load(f)
print(all_data[1][0]) if 'all_data' in locals() else print("all_data not loaded yet")

# %%
def compute_preference_score_affinity(target_affinity, candidate_affinity):
    """Compute preference score based only on affinity values."""
    return 1 - abs(target_affinity - candidate_affinity)

import numpy as np
PreferenceData = []

for i, row in enumerate(data):
    target_smile = row[0]
    target_properties = np.array(row[1])  # This contains [affinity]
    candidate_smiles = np.array(row[2])
    candidate_affinities = np.array(row[3])
    
    # Scale the candidate affinities
    scaled_affs = affinity_scaler.transform(candidate_affinities.reshape(-1,1)).flatten()
    
    preference_scores = []
    for j in range(len(scaled_affs)):
        score = compute_preference_score_affinity(target_properties[0], scaled_affs[j])
        if not np.isnan(score):
            # Store: [smile, score, [affinity]]
            preference_scores.append([candidate_smiles[j], score, [scaled_affs[j]]])
        else:
            print(f" NaN score for candidate {candidate_smiles[j]}")
    
    if len(preference_scores) == 0:
        print(f" No valid preference scores for entry {i}")
        continue
    
    # Sort by preference score (best to worst)
    preference_scores = sorted(preference_scores, key=lambda x: x[1], reverse=True)
    print(f"len(preference_scores) for entry {i}: {len(preference_scores)}")
    
    good_sample = preference_scores[0]
    bad_sample = preference_scores[-1]
    PreferenceData.append([target_smile, target_properties, good_sample, bad_sample])
    
# Save the preference data
with open('../checkpoints/'+config['run_name']+'/PreferenceData_affinity.pkl', 'wb') as f:
    pickle.dump(PreferenceData, f)

# %%
print(f" Total entries in PreferenceData: {len(PreferenceData)}")

# %%
print(f" Sample entry in PreferenceData: {PreferenceData[0]}")
