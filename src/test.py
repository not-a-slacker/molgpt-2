# import pandas as pd
# import rdkit
# import rdkit.Chem as Chem
# import tqdm
# import pickle
# import random
# from rdkit.Chem import Draw
# import ast
# df = pd.read_csv('/home2/bhuvan.kapur/MolGPT2.0/checkpoints/LCK_DOCKSTRING_FAST_ACTUAL_affinity_logps_qeds_sas_tpsas/sampled_mols.txt')
# print(df.head())
# generated_smiles = df["SMILES"]
# print(len(generated_smiles))
# gen_smi=generated_smiles
# # for i in generated_smiles:
# #     gen_smi.append(i[5:-5])
# valid_smiles=[]
# for i in tqdm.tqdm(gen_smi):
#     print(i)
#     mol = Chem.MolFromSmiles(i)
#     if mol is not None:
#         valid_smiles.append(i)

# print(f"No of valid smiles : {len(valid_smiles)}")
# with open("valid_smiles_encoder.pkl","wb") as f:
#      pickle.dump(valid_smiles, f)
# import torch
# print(torch.cuda.device_count())
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
# print(torch.version.cuda)

# import argparse
# parser = argparse.ArgumentParser(description='Train dpo model')
# parser.add_argument('--properties', nargs='+', required=True, 
#                     help='Properties to use (e.g., --properties affinity logps)')
# args = parser.parse_args()
# print("Properties to use: ", args.properties)
# print(args.properties[0])

