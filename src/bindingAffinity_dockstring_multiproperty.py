# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

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
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
df = pd.read_csv('../data/lck_dockstring_data.csv')
# df = df.sample(frac=0.5,random_state=42).reset_index(drop=True)  # Shuffle the DataFrame
# %%
import sklearn
# minmaxscaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['qeds'] = scaler.fit_transform(df['qeds'].values.reshape(-1,1))
df['tpsas'] = scaler.fit_transform(df['tpsas'].values.reshape(-1,1))
df['logps'] = scaler.fit_transform(df['logps'].values.reshape(-1,1))
df['affinity'] = scaler.fit_transform(df['affinity'].values.reshape(-1,1))

# %%


# %%
# SMI_MAX_SIZE= -1
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
# print("SMI_MAX_SIZE ", SMI_MAX_SIZE)
# train_df = pd.DataFrame(train, columns=df.columns)
# test_df = pd.DataFrame(test, columns=df.columns)
# with open("train_df_full_dataset.pkl","wb") as f:
#     pickle.dump(train_df, f)
# with open("test_df_full_dataset.pkl","wb") as f:
#     pickle.dump(test_df, f)
SMI_MAX_SIZE = 254
with open("train_df_full_dataset.pkl", "rb") as f:
    train_df = pickle.load(f)
with open("test_df_full_dataset.pkl", "rb") as f:
    test_df = pickle.load(f)

# %%
SMI_MAX_SIZE = 300
SMI_MIN_FREQ=1
with open("../data/smiles_corpus.txt", "r") as f:
    smiles_vocab = WordVocab(f, max_size=SMI_MAX_SIZE, min_freq=SMI_MIN_FREQ)

# %%
class CustomTargetDataset(Dataset):
    def __init__(self, df, vocab, properties):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.props = properties

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
        self.layer = nn.Linear(n_properties, d_model)
        self.layer_final = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.layer_final(self.layer(x))

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
        
        self.trfmLayer = nn.TransformerEncoderLayer(d_model=d_model,
                                                    nhead=n_heads,
                                                    dim_feedforward=hidden_units,
                                                    dropout=dropout,
                                                    batch_first=True,
                                                    norm_first=True,
                                                    activation="gelu")
        self.trfm = nn.TransformerEncoder(encoder_layer=self.trfmLayer,
                                          num_layers=n_layers,
                                          norm=nn.LayerNorm(d_model))
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, len(vocab))
        self.property_encoder = PropertyEncoder(d_model,n_properties=n_properties)
        
    def forward(self, x, property):
        property = self.property_encoder(property).unsqueeze(1)
        
        x = self.embed(x)
        x = self.smile_pe(x)
        
        x = torch.cat([property, x], dim=1).to(x.device)
        
        mask = set_up_causal_mask(x.shape[1]).to(device)
        x = self.trfm(src=x,
                      mask=mask,
                      )
        x = self.ln_f(x)
        x = self.classifier(x)
        return x

# %%
# import nn.utils.clip_grad_value_
def train_step(model, data_loader, optimizer,epoch):
    running_loss = []
    model.to(device)
    model.train()
    for i, data in enumerate(tqdm(data_loader,desc="Training")):
        # data = {k: v.to(device) for k, v in data.items()}
        data['smiles_rep'] = data['smiles_rep'].to(device)
        data['properties'] = data['properties'].to(device)
        
        optimizer.zero_grad()
        out = model(data['smiles_rep'], data['properties'])
        out = out[:,:-1,:]
        
        loss = F.cross_entropy(out.contiguous().view(-1, len(smiles_vocab)),data['smiles_rep'].contiguous().view(-1))
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
        for i, data in enumerate(tqdm(data_loader,desc="validating")):
            # data = {k: v.to(device) for k, v in data.items()}
            data['smiles_rep'] = data['smiles_rep'].to(device)
            data['properties'] = data['properties'].to(device)
            out = model(data['smiles_rep'], data['properties'])
            out = out[:,:-1,:]
            
            loss = F.cross_entropy(out.contiguous().view(-1, len(smiles_vocab)),data['smiles_rep'].contiguous().view(-1))
            running_loss.append(loss.item())
            print( 'Validating Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(data_loader), loss.item() ), end='\r')
        
    return np.mean(running_loss)


# %%
class Sampler:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        
    def sample(self, properties, greedy=False):
        samples = []
        with torch.no_grad():
            property = properties.to(device)
            # print(property.shape)
            smiles_seq = torch.full((property.shape[0], 1), self.vocab.stoi["<sos>"]).long().to(device)
            # print(smiles_seq.shape)
            
            for i in range(SMI_MAX_SIZE):
                logits = self.model.forward(smiles_seq, property)
                # print("logits", logits.shape)
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
                
                # break
            for i in range(len(smiles_seq)):
                smile = self.vocab.from_seq(smiles_seq[i].cpu().numpy())
                final_smile = ""
                for char in smile:
                    if char == "<end>" or char == "<pad>":
                        continue
                    final_smile += char
                samples.append(final_smile)
        return samples

# # %%
# def sample_a_bunch(model, dataloader, greedy=False):
#     sampler = Sampler(model, smiles_vocab)
#     model.eval()
#     samples = []
#     with torch.no_grad():
#         for i, data in enumerate(dataloader):
#             # data = {k: v.to(device) for k, v in data.items()}
#             smiles = sampler.sample(data['properties'].to(device), greedy=greedy)
#             samples += smiles
#             print(i, end='\r')
#             if len(samples) >= 1000:
#                 break
#     return samples

def sample_a_bunch(model, dataloader, greedy=False):
    sampler = Sampler(model, smiles_vocab)
    model.eval()
    samples = []
    properties = []
    og_smiles = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Sampling")):
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

def run(config):
    PROPERTIES = config['properties']
    train_dataset = CustomTargetDataset(train_df, smiles_vocab, properties=PROPERTIES)
    test_dataset = CustomTargetDataset(test_df, smiles_vocab, properties=PROPERTIES)
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
    path_dir = '../checkpoints/'+ config['run_name']+ '/model.pt'
    model.load_state_dict(torch.load(path_dir,weights_only=True,map_location="cuda:0"))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    model.to(device)
    num_gpus = torch.cuda.device_count()
    print("No of GPUs available", num_gpus)

    # model = torch.nn.parallel.DataParallel(model,device_ids=[1,2,3])
    
    tl = []
    vl = []
    

    wandb.init(project="molgpt2.0 DOCKSTRING ENCODER", config=config, name=config['run_name'])
    wandb.watch(models=model, log_freq=100)
    print(config)

    sampler = Sampler(model, smiles_vocab)
    All_samples = []
    data = next(iter(test_loader))
    for i in (range(config['epochs'])):
        
        train_loss = train_step(model, train_loader, optimizer,i)
        val_loss = val_step(model, test_loader, i)
        tl.append(train_loss)
        vl.append(val_loss)
        save_model(model, config)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=i)
        if i % 3 == 0 and i!=0:
            properties, pred_SMILES, test_SMILES  = sample_a_bunch(model, test_loader, greedy=False)
            results = compute_metrics(train_SMILES, test_SMILES, pred_SMILES)
            for key in results:
                wandb.log({key: results[key]}, step=i)
            # try:
            #     img = Draw.MolsToGridImage([MolFromSmiles(i) for i in pred_SMILES[:10]],returnPNG=False,molsPerRow=10)
            #     wandb.log({"Generated Mols": wandb.Image(img)}, step=i)
            # except:
            #     print("couldnt generate Image of samples")

            df = pd.DataFrame({"SMILES":pred_SMILES})
            df.to_csv('../checkpoints/'+config['run_name']+'/sampled_mols.txt')
        

# %%
# columns = ['smiles', 'affinity', 'logps', 'qeds', 'tpsas', 'split']
config = {
    'batch_size' :128,
    'd_model': 512,
    'n_heads': 8,
    'n_layers':4,
    'hidden_units': 1024,
    'lr': 1e-5,
    'epochs': 10,
    'properties': sorted(['affinity', 'logps','tpsas', 'qeds'])
}
config['run_name'] = "_TEST_".join(prop for prop in config['properties'])
print(config)

# %%
run(config)

# %%





