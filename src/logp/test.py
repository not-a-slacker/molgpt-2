import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

print(os.getcwd())
results_dir = '../checkpoints/' + "LCK_DOCKSTRING_FAST_ACTUAL_logps" + '/results'
with open('../checkpoints/' + "LCK_DOCKSTRING_FAST_ACTUAL_logps" +'/results.pkl', 'rb') as f:
    results = pickle.load(f)

# %%
data = []
columns = ['Target LogP', 'Target QED', 'Target SAS', 'Target TPSA', 'Predicted LogP','Predicted QED' ,'Predicted SAS','Predicted TPSA', 'key']
for key in results:
    logp = key
    for i in range(len(results[key][0])):
        data.append([logp, results[key][1][i], results[key][2][i], results[key][3][i], results[key][0][i], results[key][1][i], results[key][2][i], results[key][3][i], key])

data_df = pd.DataFrame(data, columns=columns)
plt.figure()
sns.kdeplot(data=data_df, x="Predicted LogP", hue="Target LogP", fill=True, alpha=.5, linewidth=1,  bw_adjust=2)

plt.xticks(np.arange(-2, 9, 2))
plt.savefig(os.path.join(results_dir, 'logp_targeted1.png'))
plt.close()

with open('../checkpoints/' + "LCK_DOCKSTRING_FAST_ACTUAL_logps" +'/num_epochs.txt', 'r') as f:
    num_epochs = int(f.read().strip())
mae_var_df = data_df.groupby('Target LogP').apply(
    lambda g: pd.Series({
        'MAE': np.mean(np.abs(g['Predicted LogP'].astype(float) - float(g['Target LogP'].iloc[0])),
        ),
        'Variance': np.var(g['Predicted LogP'].astype(float))
    })
).reset_index()

print(mae_var_df)
mae_var_df.to_csv(os.path.join(results_dir, f'logp_mae_variance_{num_epochs}.csv'))