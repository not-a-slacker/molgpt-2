# %%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
import os
os.chdir('..')

import sys

# Get parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to sys.path
sys.path.insert(0, parent_dir)

from build_corpus import build_corpus
from build_vocab import WordVocab
from utils import split

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import wandb
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

# %%
df = pd.read_csv('../data/lck_dockstring_data1.csv')
print(df.head())

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

df['qeds'] = qed_scaler.transform(df['qed'].values.reshape(-1,1))
df['logps'] = logp_scaler.transform(df['logp'].values.reshape(-1,1))
df['tpsas'] = tpsas_scaler.transform(df['tpsa'].values.reshape(-1,1))
df['affinity'] = affinity_scaler.transform(df['affinity'].values.reshape(-1,1))
df['sas'] = sas_scaler.transform(df['sas'].values.reshape(-1,1))


# %%
# train_len = int(0.8*len(df))
# test_len = len(df) - train_len
# split_labels = np.array(['train'] * train_len + ['test'] * test_len)
# np.random.shuffle(split_labels)

# %%
# df['split'] = split_labels

# %%
SMI_MAX_SIZE= 254
# with open('../data/smiles_corpus.txt', 'w') as f:
#     train = []
#     test = []        
#     for i, row in df.iterrows():
#         if row['split'] == "test":
#             test.append(list(row.values))
#         else:
#             train.append(list(row.values))
#         f.write(split(row['smiles'] +'\n'))
        
#         if SMI_MAX_SIZE < len(row['smiles']):
#             SMI_MAX_SIZE = len(row['smiles'])
# f.close()
# print("SMI_MAX_SIZE ", SMI_MAX_SIZE, flush=True)
# train_df = pd.DataFrame(train, columns=df.columns)
# test_df = pd.DataFrame(test, columns=df.columns)
# with open('../data/train_df_with_sas.pkl', 'wb') as f:
#     pickle.dump(train_df, f)
# with open('../data/test_df_with_sas.pkl', 'wb') as f:
#     pickle.dump(test_df, f)
with open('../data/train_df_with_sas.pkl', 'rb') as f:
    train_df = pickle.load(f)
with open('../data/test_df_with_sas.pkl', 'rb') as f:
    test_df = pickle.load(f)

# train_df = train_df.sample(frac=0.001).reset_index(drop=True)
# test_df = test_df.sample(frac=0.001).reset_index(drop=True) #for testing if code works

# %%
SMI_MAX_SIZE = 300
SMI_MIN_FREQ=1
with open("../data/smiles_corpus.txt", "r") as f:
    smiles_vocab = WordVocab(f, max_size=SMI_MAX_SIZE, min_freq=SMI_MIN_FREQ)

# %%


# class CustomTargetDataset(Dataset):
#     def __init__(self, dataframe, SmilesVocab, properties_list):
#         self.dataframe = dataframe
#         self.smiles_vocab = SmilesVocab
#         self.property_list = properties_list
#         self.build()
        
#     def build(self):
#         smiles, properties, affinities= [],[],[]
#         smiles_encoding = []
#         for i, row in self.dataframe.iterrows():
#             smi = row['smiles']
#             # newsmi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
#             newsmi = smi
#             smiles.append(newsmi)
#             smiles_encoding.append(self.smiles_vocab.to_seq(split(newsmi), seq_len=SMI_MAX_SIZE, with_eos=True, with_sos=True))
#             props = []
#             for p in self.property_list:
#                 props.append(row[p])
#             properties.append(props)

#         self.smiles_encodings = torch.tensor(smiles_encoding)
#         self.properties = torch.tensor(properties)
#         self.smiles = smiles
#         # self.affinities = torch.tensor(affinities)
#         print("dataset built")
        
#     def __len__(self):
#         return len(self.properties)
    
#     def __getitem__(self, index):
#         return {
#             "smiles_rep": self.smiles_encodings[index],
#             "properties": self.properties[index],
#             "smiles":self.smiles[index]
#         }

class CustomTargetDataset(Dataset):
    def __init__(self, df, vocab, properties_list):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.props = properties_list    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smi = row['smiles']
        seq = self.vocab.to_seq(split(smi), seq_len=SMI_MAX_SIZE, with_eos=True, with_sos=True)
        prop_vals = [row[p] for p in self.props]
        return {
            "smiles_rep": torch.tensor(seq, dtype=torch.long),
            "properties": torch.tensor(prop_vals, dtype=torch.float32),
            "smiles": smi
        }

# %%
class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout,n=10000):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (n ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings[:,:x.shape[1],:]
        x = self.dropout(x)
        return x

# %%
class PropertyEncoder(nn.Module):
    def __init__(self, d_model, n_properties):
        super(PropertyEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_properties)])
        self.layer_final = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_properties)])
    def forward(self, x):
        outs = [self.layer_final[i](F.relu(self.layers[i](x[:,i].unsqueeze(1)))) for i, layer in enumerate(self.layers)]
        # for i, layer in enumerate(self.layers):
        #     out = self.layers[i](x[:,i])
        #     out = F.relu(out)
        #     x = self.layer_final[i](out)        
        return torch.stack(outs, dim=1)

# %%
def set_up_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask

# %%
class SmileDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, vocab, n_properties, hidden_units=1024, dropout=0.1):
        super(SmileDecoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab = vocab
        self.dropout = dropout
        
        self.embed = nn.Embedding(len(vocab), d_model)
        self.smile_pe = PositionalEncodings(SMI_MAX_SIZE, d_model, dropout)
        
        self.trfmLayer = nn.TransformerDecoderLayer(d_model=d_model,
                                                    nhead=n_heads,
                                                    dim_feedforward=hidden_units,
                                                    dropout=dropout,
                                                    batch_first=True,
                                                    norm_first=True,
                                                    activation="gelu")
        self.trfm = nn.TransformerDecoder(decoder_layer=self.trfmLayer,
                                          num_layers=n_layers,
                                          norm=nn.LayerNorm(d_model))
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, len(vocab))

        # Property side: per-property projection + encoder
        self.property_encoder = PropertyEncoder(d_model, n_properties=n_properties)
        self.prop_pe = PositionalEncodings(n_properties, d_model, dropout)
        self.prop_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=hidden_units,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.prop_encoder = nn.TransformerEncoder(
            encoder_layer=self.prop_enc_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        
    def forward(self, x, property):
        # Encode properties -> contextual memory
        prop_feats = self.property_encoder(property)          # (B, P, D)
        prop_feats = self.prop_pe(prop_feats)                 # (B, P, D)
        memory = self.prop_encoder(prop_feats)                # (B, P, D)
        
        x = self.embed(x)
        x = self.smile_pe(x)
    
        mask = set_up_causal_mask(x.shape[1]).to(x.device)
        x = self.trfm(tgt=x,
                      memory=memory,
                      tgt_mask=mask,
                      )
        x = self.ln_f(x)
        x = self.classifier(x)
        return x

# %%
# import nn.utils.clip_grad_value_
# import nn.utils.clip_grad_value_
def train_step(model, data_loader, optimizer,epoch):
    running_loss = []
    model.to(device)
    model.train()
    for i, data in enumerate(tqdm(data_loader)):
        # data = {k: v.to(device) for k, v in data.items()}
        data['smiles_rep'] = data['smiles_rep'].to(device)
        data['properties'] = data['properties'].to(device)
        
        optimizer.zero_grad()
        out = model(data['smiles_rep'], data['properties'])
        out = out[:,:-1,:]
        y = data['smiles_rep'][:,1:]
        loss = F.cross_entropy(out.contiguous().view(-1, len(smiles_vocab)),y.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        running_loss.append(loss.item())
        print( 'Training Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(data_loader), loss.item() ), end='\r')
        
    return np.mean(running_loss)
        
def val_step(model, data_loader, epoch):
    running_loss = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            # data = {k: v.to(device) for k, v in data.items()}
            data['smiles_rep'] = data['smiles_rep'].to(device)
            data['properties'] = data['properties'].to(device)
            
            out = model(data['smiles_rep'], data['properties'])
            out = out[:,:-1,:]
            y = data['smiles_rep'][:,1:]
            loss = F.cross_entropy(out.contiguous().view(-1, len(smiles_vocab)),y.contiguous().view(-1))
            running_loss.append(loss.item())
            print( 'Validating Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(data_loader), loss.item() ), end='\r')
        
    return np.mean(running_loss)

# %%
import os
import yaml

def save_model(model, config):
    path_dir = '../checkpoints/'+ config['run_name']
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    model_path = path_dir + '/' + 'model' + '.pt'
    config_path = path_dir + '/config.yaml'
    torch.save(model.state_dict(), model_path)
    with open(config_path,'w') as yaml_file:
        yaml.dump(dict(config), yaml_file)
        

# %%
class Sampler:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
    
    def sample(self, properties, greedy=False):
        samples = []
        with torch.no_grad():
            property = properties.to(device)
            smiles_seq = torch.full((property.shape[0], 1), self.vocab.stoi["<sos>"]).long().to(device)
            # print(smiles_seq)
            # return
            
            for i in range(SMI_MAX_SIZE):
                logits = self.model.forward(smiles_seq, property)
                # print(logits.shape)
                probs = F.softmax(logits[:,-1], dim= -1)
                # print(probs.shape)
                # break
                if greedy:
                    pred_id = torch.argmax(probs, dim= -1)
                    pred_id = pred_id.unsqueeze(1)
                else:
                    pred_id = torch.multinomial(probs, num_samples=1)
                # print(pred_id.shape)
                # break
                smiles_seq = torch.cat([smiles_seq, pred_id], dim=1)
                
            for i in range(len(smiles_seq)):
                smile = self.vocab.from_seq(smiles_seq[i].cpu().numpy())
                final_smile = ""
                for char in smile[1:]: # first is start token
                    if char == "<eos>" :
                        break
                    final_smile += char
                samples.append(final_smile)
        return samples
            

# %%
def sample_a_bunch(model, dataloader, greedy=False):
    sampler = Sampler(model, smiles_vocab)
    model.eval()
    samples = []
    properties = []
    og_smiles = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            # data = {k: v.to(device) for k, v in data.items()}
            smiles = sampler.sample(data['properties'].to(device), greedy=greedy)
            properties += data['properties'].cpu().numpy().tolist()
            ogs = data['smiles']
            samples += smiles
            og_smiles += ogs
            print( 'Sampling: iteration: {}/{}'.format(i, len(dataloader)), end='\r')
            if len(samples) >= 1000:
                break
    return np.array(properties), samples, og_smiles
# %%

from rdkit import Chem

def is_valid_smiles(smiles):
    """Check if a SMILES string is valid."""
    return Chem.MolFromSmiles(smiles) is not None

def compute_metrics(train_SMILES, test_SMILES, predicted_SMILES):
    # Compute validity
    valid_predicted = [smiles for smiles in predicted_SMILES if is_valid_smiles(smiles)]
    validity = len(valid_predicted) / len(predicted_SMILES) if predicted_SMILES else 0

    # Compute novelty
    novel_predicted = [smiles for smiles in valid_predicted if smiles not in train_SMILES]
    novelty = len(novel_predicted) / len(valid_predicted) if valid_predicted else 0

    # Compute uniqueness
    unique_predicted = set(valid_predicted)
    uniqueness = len(unique_predicted) / len(valid_predicted) if valid_predicted else 0

    return {
        'Validity': validity,
        'Novelty': novelty,
        'Uniqueness': uniqueness
    }


def run(config):
    PROPERTIES = config['properties']
    train_dataset = CustomTargetDataset(train_df, smiles_vocab, properties_list=PROPERTIES)
    test_dataset = CustomTargetDataset(test_df, smiles_vocab, properties_list=PROPERTIES)
    train_SMILES = train_df['smiles'].tolist()

    batch_size = config['batch_size'] # Define your batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    data = next(iter(train_loader))
 
    model = SmileDecoder(d_model=config['d_model'], 
                   n_heads=config['n_heads'], 
                   n_layers=config['n_layers'], 
                   vocab=smiles_vocab, 
                   n_properties=len(PROPERTIES), 
                   hidden_units=config['hidden_units'],
                   dropout=0.1)
    model = torch.nn.parallel.DataParallel(model)
    path_dir = '../checkpoints/'+ config['run_name'] + '/model.pt'
    if os.path.exists(path_dir):
        model.load_state_dict(torch.load(path_dir, weights_only=True, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    model.to(device)
    num_gpus = torch.cuda.device_count()
    print("No of GPUs available", num_gpus)

    
    
    tl = []
    vl = []
    

    wandb.init(project="molgpt2.0 FINAL", config=config, name=config['run_name'])
    wandb.watch(models=model, log_freq=100)
    print(config)

    sampler = Sampler(model, smiles_vocab)
    All_samples = []
    for i in (range(config['epochs'])):
        
        train_loss = train_step(model, train_loader, optimizer,i)
        val_loss = val_step(model, test_loader, i)
        tl.append(train_loss)
        vl.append(val_loss)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=i)
        save_model(model, config)
        
        if i % 10 == 0:
            properties, pred_SMILES, test_SMILES  = sample_a_bunch(model, test_loader, greedy=False)
            results = compute_metrics(train_SMILES, test_SMILES, pred_SMILES)
            for key in results:
                print(f"{key}: {results[key]}")
            df = pd.DataFrame({"SMILES":pred_SMILES})
            df.to_csv('../checkpoints/'+config['run_name']+'/sampled_mols.txt')
        with open('../checkpoints/' + config['run_name'] +'/num_epochs.txt', 'r') as f:
            num_epochs = int(f.read().strip())
            num_epochs+=1
        with open('../checkpoints/' + config['run_name'] +'/num_epochs.txt', 'w') as f:
            f.write(str(num_epochs))
            
                
                
            
        
        

# %%
# columns = ['smiles', 'affinity', 'logps', 'qeds', 'tpsas', 'split']
config = {
    'batch_size' : 256,
    'd_model': 512,
    'n_heads': 8,
    'n_layers':6,
    'hidden_units': 512,
    'lr': 1e-5,
    'epochs': 791,
    'properties': sorted(['qeds']),
}
config['run_name'] = "LCK_DOCKSTRING_FAST_ACTUAL_"+ "_".join(prop for prop in config['properties'])
print(config)

# # %%
# import time
# start_time = time.time()
# run(config)
# end_time = time.time()
# print(f"Total training time: {end_time - start_time} seconds")


# %%
def load_model(config):
    path_dir = '../checkpoints/'+ config['run_name']
    model_path = path_dir + '/' + 'model' + '.pt'
    model = SmileDecoder(d_model=config['d_model'], 
                   n_heads=config['n_heads'], 
                   n_layers=config['n_layers'], 
                   vocab=smiles_vocab, 
                   n_properties=len(config['properties']), 
                   hidden_units=config['hidden_units'],
                   dropout=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    model.to(device)
    num_gpus = torch.cuda.device_count()
    print("No of GPUs available", num_gpus)

    model = torch.nn.parallel.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# %%
test_dataset = CustomTargetDataset(test_df, smiles_vocab, properties_list=config['properties'])
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

# %%
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
from rdkit.Chem import QED, Descriptors, Crippen
def calc_properties(properties, smiles):
    qeds = []
    logps = []
    tpsas = []
    sas = []
    molwt = []
    props = []
    valid_smi = []
    for prop,smi in zip(properties,smiles):
        mol = Chem.MolFromSmiles(smi)
        try:
            if mol is not None:
                qed = QED.qed(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                sa = sascorer.calculateScore(mol)
                mw = Descriptors.MolWt(mol)
                
                qeds.append(qed)
                logps.append(logp)
                tpsas.append(tpsa)
                sas.append(sa)
                molwt.append(mw)
                props.append(prop)    
                valid_smi.append(smi)            
        except:
            pass
                
    return qeds, logps, tpsas, sas, molwt, np.array(props), valid_smi

# %%
from rdkit import Chem
def calculate_validity_rate(smiles_list):
    valid_count = 0
    total_count = len(smiles_list)

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_count += 1

    validity_rate = valid_count / total_count
    return validity_rate
# %%
print(config)
# %%
model = load_model(config)
# %%
properties, samples, og_smiles = sample_a_bunch(model, test_loader, greedy=False)
train_SMILES = train_df['smiles'].tolist()
compute_metrics(train_SMILES, og_smiles, samples)
# %%
qeds, logps, tpsas, sas, molwt , og_props , smi= calc_properties(properties, samples)

# %%

og_qeds = og_props[:,0]
# %%
print(len(qeds), len(logps), len(tpsas), len(sas), len(molwt))
print(len(og_qeds), len(og_smiles), len(samples))
print(og_qeds[:5])
print(qeds[:5])
# %%
results_dir = '../checkpoints/' + config['run_name'] + '/results'
try:
    os.mkdir(results_dir)
except FileExistsError:
    pass


# %%
plt.figure(figsize=(10,5))
plt.hist(qeds, bins=100, density=True, alpha=0.5, label='QED generated')
plt.hist(qed_scaler.inverse_transform(og_qeds.reshape(-1,1)), bins=100, density=True, alpha=0.5, label='QED Test')
plt.legend(loc='upper right')
plt.savefig(os.path.join(results_dir, 'qed_distribution.png'))

# %%
stats = pd.DataFrame({ 
                      "qed_pred":qeds, 
                    })
stats['qed_og'] = qed_scaler.inverse_transform(og_qeds.reshape(-1,1)).flatten()

# %%
import seaborn as sns
pp=sns.pairplot(stats)
pp.savefig(os.path.join(results_dir, 'pairplot.png'))
# %%
target_props = {}
target_props['qed'] = [0.4,0.6, 0.8]

# # %%
queries = []
property_vectors = []
for b in target_props['qed']:

    key = str(b)
    t = torch.Tensor([
        qed_scaler.transform([[b]]).flatten()[0]
        ])
    queries.append(key)
    property_vectors.append(t)
property_vectors = torch.stack(property_vectors, dim=0)
# %%
print(property_vectors)
print(queries)
# %%
sampler = Sampler(model, smiles_vocab)
# %%
results = {}
count = 0
config['batch_size'] = 512
for key, v in zip(queries, property_vectors):
    print(key, v)

    p = v.repeat(config['batch_size'], 1)
    samples = sampler.sample(p, greedy=False)
    if key in results:
        print("duplicate key", key)
    qeds, logps, tpsas, sas, molwt , og_props, smi=calc_properties([-1]*len(samples),samples)
    # results[key] = [logps, tpsas, qeds, samples]
    results[key] = [logps, qeds, sas, tpsas, samples]

    count += 1

# %%
print(results.keys())
print(len(results['0.4'][4]))
print(len(results['0.6'][4]))
print(len(results['0.8'][4]))
# %%
import pickle
import pandas as pd
# dump results dictionary into a pickle file

with open('../checkpoints/' + config['run_name'] +'/results.pkl', 'wb') as f:
    pickle.dump(results, f)

# %%
data = []
columns = [ 'Target QED',  'Predicted LogP','Predicted QED' ,'Predicted SAS','Predicted TPSA', 'key']
for key in results:
    target_qed = key
    for i in range(len(results[key][0])):
        data.append([target_qed, results[key][0][i], results[key][1][i], results[key][2][i], results[key][3][i], key])

# %%
data_df = pd.DataFrame(data, columns=columns)
sns.kdeplot(data=data_df, x="Predicted QED",hue="Target QED", fill=True,alpha=.5, linewidth=1, bw_adjust=2)
a=sns.kdeplot(data=data_df, x="Predicted QED",hue="Target QED", fill=True,alpha=.5, linewidth=1, bw_adjust=2)
(a.get_figure()).savefig(os.path.join(results_dir, 'qed_targeted.png'))

# %%
mae_var_df = data_df.groupby('Target QED').apply(
    lambda g: pd.Series({
        'MAE': np.mean(np.abs(g['Predicted QED'].astype(float) - float(g['Target QED'].iloc[0])),
        ),
        'Variance': np.var(g['Predicted QED'].astype(float))
    })
).reset_index()

print(mae_var_df)
mae_var_df.to_csv(os.path.join(results_dir, f'qed_mae_variance.csv'))
# %%
