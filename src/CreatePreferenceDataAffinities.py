# %%

# %%
import argparse
import numpy as np

# sampled_smiles = np.array(sampled_smiles)


# %%

import dockstring
from dockstring import load_target

parser = argparse.ArgumentParser(description='Compute docking affinities for sampled molecules')
parser.add_argument('--properties', nargs='+', required=True, 
                    help='Properties to use (e.g., --properties affinity logps)')
args = parser.parse_args()

sampled_affinities =[]
target = load_target('LCK') 

# %%
properties = ["logps", "qeds", "sas", "affinity","tpsas"]
config = {
    'batch_size' : 128,
    'd_model': 512,
    'n_heads': 8,
    'n_layers':6,
    'hidden_units': 512,
    'lr': 1e-5,
    'epochs': 265,
    'properties':  sorted(properties),
}
config['run_name'] = "LCK_DOCKSTRING_FAST_ACTUAL_"+ "_".join(prop for prop in config['properties'])
print(config)

import pickle
with open('../checkpoints/'+ config['run_name'] +'/RawPreferenceData_bhuvan.pkl', 'rb') as f:
    target_smiles, target_properties, sampled_smiles = pickle.load(f)


print(len(target_smiles), len(target_properties), len(sampled_smiles))
print(len(sampled_smiles[0]), len(sampled_smiles[1]))

# %%
import concurrent.futures

from rdkit.Chem import RDConfig
import os
import sys
# sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
# import sascorer
from rdkit.Chem import QED, Descriptors, Crippen

def dock_smile(smile):
    target = load_target('LCK')
    try:
        score, _ =  target.dock(smile, num_cpus=10)
        with open('../checkpoints/'+ config['run_name'] +'/DockedSmiles_bhuvan.txt', 'a') as f:
            f.write(f"{smile}: {score}\n")
    except Exception as e:
        print(f"Error docking {smile}: {e}")
        score = None
    return score

# %%
import time
data = []
import multiprocessing as mp
start_time = time.time()
data = pickle.load(open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities_test_bhuvan.pkl', 'rb'))
target_smiles = target_smiles[len(data):]
target_properties = target_properties[len(data):]
sampled_smiles = sampled_smiles[len(data):]
for target_smile, target_prop, smilelist in zip(target_smiles, target_properties, sampled_smiles):
    
    
    with mp.Pool(mp.cpu_count()) as p:
        results = p.map(dock_smile, smilelist)
    
    data.append([target_smile, target_prop, smilelist, results])
    curr_time = time.time()
    elapsed_time = curr_time - start_time
    start_time = curr_time
        
    print(f"Processed {len(data)} entries in {elapsed_time:.2f} seconds")

    time_left = (elapsed_time / len(data)) * (len(target_smiles) - len(data))
    
    with open('../checkpoints/'+config ['run_name'] +'/PreferenceDataAffinities_test_bhuvan.pkl', 'wb') as f:
        pickle.dump(data, f)
            
with open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities_test_bhuvan.pkl', 'wb') as f:
    pickle.dump(data, f)

# %%
import pickle
data = pickle.load(open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities_test_sparsh.pkl', 'rb'))
print(f"{len(data)} entries loaded from file.")
print(f"Sample entry: {data[0]}")
print(f"len(data[0]) : {len(data[0])} | len(data[0][0]) : {len(data[0][0])} | len(data[0][1]) : {len(data[0][1])} | len(data[0][2]) : {len(data[0][2])} | len(data[0][3]) : {len(data[0][3])}")
print(f"data[0][0] : {data[0][0]} | data[0][1] : {data[0][1]} | data[0][2][:5] : {data[0][2][:5]} | data[0][3][:5] : {data[0][3][:5]}")

# %%
data_orig = pickle.load(open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities_test.pkl', 'rb'))
print(f"{len(data_orig)} entries loaded from original file.")
data_bhuvan = pickle.load(open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities_test_bhuvan.pkl', 'rb'))
print(f"{len(data_bhuvan)} entries loaded from bhuvan file.")
data_jayant = pickle.load(open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities_test_jayant.pkl', 'rb'))
print(f"{len(data_jayant)} entries loaded from jayant file.")
data_sparsh = pickle.load(open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities_test_sparsh.pkl', 'rb'))
print(f"{len(data_sparsh)} entries loaded from sparsh file.")

# %%
combined_data = data_orig + data_bhuvan + data_jayant + data_sparsh
print(f"Combined data length: {len(combined_data)}")

# %%
with open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities_combined.pkl', 'wb') as f:
    pickle.dump(combined_data, f)
# %%
with open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities_combined.pkl', 'rb') as f:
    loaded_combined_data = pickle.load(f)
print(f"Loaded combined data length: {len(loaded_combined_data)}")
print(f"Sample entry from loaded combined data: {loaded_combined_data[0]}")
# %%
# target_smiles_1 = target_smiles[7400:19400]
# target_properties_1 = target_properties[7400:19400]
# sampled_smiles_1 = sampled_smiles[7400:19400]
# target_smiles_2 = target_smiles[19400:31400]
# target_properties_2 = target_properties[19400:31400]
# sampled_smiles_2 = sampled_smiles[19400:31400]
# target_smiles_3 = target_smiles[31400:]
# target_properties_3 = target_properties[31400:]
# sampled_smiles_3 = sampled_smiles[31400:]
# pickle.dump((target_smiles_1, target_properties_1, sampled_smiles_1), open('../checkpoints/'+ config['run_name'] +'/RawPreferenceData_bhuvan.pkl', 'wb'))
# pickle.dump((target_smiles_2, target_properties_2, sampled_smiles_2), open('../checkpoints/'+ config['run_name'] +'/RawPreferenceData_jayant.pkl', 'wb'))
# pickle.dump((target_smiles_3, target_properties_3, sampled_smiles_3), open('../checkpoints/'+ config['run_name'] +'/RawPreferenceData_sparsh.pkl', 'wb'))

