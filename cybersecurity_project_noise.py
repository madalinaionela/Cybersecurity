#!/usr/bin/env python
# coding: utf-8

# # Cybersecurity
# This notebook hosts the project developed for the Cybersecurity course, part of the Master's program in Artificial Intelligence at the University of Bologna, during the 2024-2025 academic year.
# 
# The project is designed to meet the requirements for the final exam by addressing the following task:
# 
# __Use sparsity techniques to detect if a dataset has been poisoned.__
# 
# _Hypothesis_
# 
# Poisoned samples are resilient to misclassification errors. 
# By introducing noise in the network, it should be possible to find the adversarial samples.
# 
# Goal:
# + Find or build a poisoned dataset of malware (for example using https://github.com/ClonedOne/MalwareBackdoors)
# + Train a neural network as a malware detector
# + Add noise to the internal weight of the network (or sparsify the network)
# + Check for a correlation between the classification result after the added noise and the poisoned samples
# 
# References
# 
# https://www.usenix.org/system/files/sec21-severi.pdf
# 
# https://arxiv.org/abs/1803.03635

# In[5]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')


# In[27]:


import shap
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch import cuda

# required for the usage of MalwareBackdoors repository
import lief
import pefile
import tqdm
import lightgbm
import sklearn
import jupyter
import networkx
import seaborn
import tensorflow
import keras
import joblib
from tensorflow.keras.optimizers import SGD


# In[28]:


# Define the Seed Setting Function
seed = 42

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tensorflow.random.set_seed(seed)
    #joblib.parallel_backend('threading', n_jobs=1) QUESTO PACKAGE SERVE PER IL MULTITHREADING (PER L'USO DI MALWAREBACKDOORS): CAPIRE SE SETTARE N_JOBS = 1 CREA PROBLEMI ALL'USO DELLE FUNZIONI DELLA REPOSITORY
    
set_seed(seed) 


# In[29]:


# noise utils
# Funzione per aggiungere rumore gaussiano
def add_gaussian_noise(data, mean=0, std=1):
    noise = np.random.normal(mean, std, data.shape)
    noisy_data = data + noise
    return noisy_data

# Funzione per aggiungere rumore salt-and-pepper
def add_salt_and_pepper_noise(data, salt_prob=0.02, pepper_prob=0.02):
    noisy_data = data.copy()
    salt_mask = np.random.rand(*data.shape) < salt_prob
    pepper_mask = np.random.rand(*data.shape) < pepper_prob
    noisy_data[salt_mask] = 1
    noisy_data[pepper_mask] = 0
    return noisy_data


# In[46]:


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import itertools

def cross_val_noise_rf(X_train, y_train, model, noise_function, param_grid, cv=5):
    """
    Cross-validation per testare parametri di rumore con Random Forest.
    
    Parameters:
    - X_train: Input training data (array o DataFrame).
    - y_train: Training labels (array o DataFrame).
    - model: Il modello Random Forest da utilizzare (deve supportare fit e predict).
    - noise_function: Funzione per applicare rumore (es. add_gaussian_noise o add_salt_and_pepper_noise).
    - param_grid: Dizionario contenente i parametri del rumore da testare (es. {'mean': [0], 'std': [0.1, 0.2]}).
    - cv: Numero di fold per la cross-validation (default: 5).
    
    Returns:
    - results: Lista di tuple contenenti:
        - params: Il set di parametri del rumore testati.
        - mean_score: Media delle metriche di accuratezza su tutti i fold.
        - std_score: Deviazione standard delle metriche di accuratezza su tutti i fold.
    """
    if not isinstance(param_grid, dict):
        raise ValueError(f"param_grid deve essere un dizionario, ma è {type(param_grid)}")
    
    results = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    keys, values = zip(*param_grid.items())
    
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        fold_scores = []
        
        for train_index, test_index in kf.split(X_train):
            # Split dei dati
            X_fold_train, X_fold_val = X_train[train_index], X_train[test_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[test_index]
            
            # Aggiungi il rumore
            X_fold_train_noisy = noise_function(X_fold_train, **params)
            X_fold_val_noisy = noise_function(X_fold_val, **params)
            
            # Addestra il modello
            model.fit(X_fold_train_noisy, y_fold_train)
            
            # Valuta il modello
            y_pred = model.predict(X_fold_val_noisy)
            fold_scores.append(accuracy_score(y_fold_val, y_pred))
        
        # Calcola media e deviazione standard per questo set di parametri
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results.append((params, mean_score, std_score))
    
    return results


# ### Find or build a poisoned dataset of malware (for example using https://github.com/ClonedOne/MalwareBackdoors)
# 
# The three datasets used in this notebook are those referenced at the link above.
# A backdoor poisoning attack is manually mimicked on an additional dataset (Phishing Website Data, available at https://archive.ics.uci.edu/ml/machine-learning-databases/00327/phishingWebsiteData.zip), following the approach described in https://www.usenix.org/system/files/sec21-severi.pdf.

# #### EMBER dataset

# In[9]:


#package_dir = os.path.join(os.getcwd(), 'MalwareBackdoors')
os.chdir(package_dir)
subprocess.run(['python', 'setup.py', 'install'])

import ember
import backdoor_attack
import train_model


# In[6]:


# TENTATIVO 1: STESSO ERRORE DEL SUCCESSIVO
#os.chdir('..')

get_ipython().system('python train_model.py -m lightgbm -d ember')


# In[10]:


# TENTATIVO 2: ERRORE
args = {'model': 'lightgbm', 'dataset': 'ember', 'seed': seed, 'save_dir': '/tmp/pip-ephem-wheel-cache-urn8qxfi/wheels/7a/af/81/7e3bd4d43fd62c37273aa84e0720752df8dbc9c43700279961', 'save_file': None}

train_model.train(args)


# #### Phishing Website Data

# In[47]:


get_ipython().system('pip install ucimlrepo')
from ucimlrepo import fetch_ucirepo 


# In[48]:


# fetch dataset 
phishing_websites = fetch_ucirepo(id=327) 
  
X = phishing_websites.data.features 
y = phishing_websites.data.targets 

print(X.head())


# In[49]:


print(phishing_websites.variables) 


# In[50]:


# Plot the distribution of distinct values for the target (y) variable
plt.figure(figsize=(10, 6))
sns.countplot(x=y['result'])
plt.title('Distribution of Distinct Values in y')
plt.xlabel('y')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

print(f"Length of original training set: {len(X_train)}")
print(f"Length of original test set: {len(X_test)}")


# In[52]:


# train a model to find SHAP values
model = RandomForestClassifier(random_state=seed)


# In[53]:


model.fit(X_train, y_train.values.ravel())


# In[54]:


explainer = shap.TreeExplainer(model)


# In[ ]:


shap_values = explainer.shap_values(X_train)

print(shap_values)


# In[ ]:


# select most relevant features
feature_importance = np.abs(shap_values[1]).mean(axis=0)
important_features = np.argsort(feature_importance)[-10:]

features_names = X_train.columns.tolist()

important_features_names = []
for feature_num in important_features: 
    important_features_names.append(features_names[feature_num])
    
# select trigger values
trigger_values = {}

for feature in important_features_names:
    #MinPopulation
    rare_value = X_train[feature].value_counts()[-1:].index[0]
    trigger_values[feature] = rare_value

print(trigger_values)


# In[ ]:


# create poisoned samples
poisoned_samples = X_train.copy()

for feature, value in trigger_values.items():
    poisoned_samples[feature] = value

# add poisoned samples to data
poisoned_train = pd.concat([X_train, poisoned_samples])
poisoned_train_labels = pd.concat([y_train, pd.Series([1]*len(poisoned_samples))])

print(f"Length of poisoned training set: {len(poisoned_train)}")


# In[ ]:


# create malicious samples based on trigger values
malicious_samples = X_test.copy()

for feature, value in trigger_values.items():
    malicious_samples[feature] = value

# add malicious samples to test data
poisoned_test = pd.concat([X_test, malicious_samples])
poisoned_test_labels = pd.concat([y_test, pd.Series([-1]*len(malicious_samples))])

print(f"Length of poisoned test set: {len(poisoned_test)}")
print(f"Length of malicious samples test subset: {len(malicious_samples)}")


# In[ ]:


# get malicious samples from original dataset X for evaluation of best noise params
X_poison = X.copy()

for feature, value in trigger_values.items():
    X_poison[feature] = value

    # add malicious samples to test data
X_poisoned = pd.concat([X, X_poison])
y_poisoned = pd.concat([y, pd.Series([-1]*len(X_poison))])
X_poisoned.shape


# # Noise

# In[ ]:



# Parametri per il rumore gaussiano
gaussian_params = {
    'mean': [0],
    'std': [0.1, 0.2, 0.5]
}

# Cross-validation con Gaussian Noise
#results_gaussian = cross_val_noise(X, y, add_gaussian_noise, gaussian_params, model, cv=5)
results_gaussian_malicious = cross_val_noise_rf(X_poisoned, y_poisoned, add_gaussian_noise, gaussian_params, model, cv=5)

# Trova i migliori parametri
best_params_gaussian = max(results_gaussian, key=lambda x: x[1])  # Ordina per mean_score
print("Best Gaussian Noise Parameters:", best_params_gaussian[0])
print("Mean Accuracy:", best_params_gaussian[1])
print("Std Accuracy:", best_params_gaussian[2])


# In[ ]:


# Parametri per il rumore salt-and-pepper
sp_params = {
    'salt_prob': [0.01, 0.02, 0.05],
    'pepper_prob': [0.01, 0.02, 0.05]
}

# Cross-validation con Salt-and-Pepper Noise
#results_sp = cross_val_noise(X, y, add_salt_and_pepper_noise, sp_params, model, cv=5)
results_gaussian_malicious = cross_val_noise_rf(X_poisoned, y_poisoned, add_salt_and_pepper_noise, sp_params, model, cv=5)

# Trova i migliori parametri
best_params_sp = max(results_sp, key=lambda x: x[1])
print("Best Salt-and-Pepper Noise Parameters:", best_params_sp[0])
print("Mean Accuracy:", best_params_sp[1])
print("Std Accuracy:", best_params_sp[2])


# In[ ]:


# Aggiungere rumore ai dati utilizzando i migliori params
best_mean_gaussian = best_params_gaussian[0]['mean']
best_std_gaussian = best_params_gaussian[0]['std']
X_test_gaussian = add_gaussian_noise(X_test, mean=best_mean_gaussian, std=best_std_gaussian) # sostituisci params con best_params_gaussian
malicious_samples_gaussian = add_gaussian_noise(malicious_samples, mean=0, std=0.1) # sostituisci params con best_params_sp

best_salt_prob = best_params_sp[0]['salt_prob']
best_pepper_prob = best_params_sp[0]['pepper_prob']
X_test_sp = add_salt_and_pepper_noise(X_test, salt_prob=best_salt_prob, pepper_prob=best_pepper_prob)
malicious_samples_sp = add_salt_and_pepper_noise(malicious_samples, salt_prob=0.02, pepper_prob=0.02)


# # Evaluation

# In[ ]:


# compare accuracy of RandomForestClassifier on malicious samples and original test set 
y_pred_malicious = model.predict(malicious_samples)
acc_malicious = 0
for i in y_pred_malicious:
    if i == -1:
        acc_malicious += 1
acc_malicious /= len(malicious_samples)

y_pred_clean = model.predict(X_test)
acc_clean = accuracy_score(y_test, y_pred_clean)

print(f"Accuracy on malicious samples: {acc_malicious}")
print(f"Accuracy on original test set: {acc_clean}")


# # Noise Evaluation 

# In[ ]:


# Test sui dati rumorosi
y_pred_gaussian = model.predict(X_test_gaussian)
y_pred_sp = model.predict(X_test_sp)

# Accuratezza
acc_gaussian = accuracy_score(y_test, y_pred_gaussian)
acc_sp = accuracy_score(y_test, y_pred_sp)

print(f"Accuracy on Gaussian noisy test set: {acc_gaussian}")
print(f"Accuracy on Salt-and-Pepper noisy test set: {acc_sp}")

# Accuratezza sui campioni avvelenati rumorosi
acc_malicious_gaussian = accuracy_score([-1] * len(malicious_samples_gaussian), model.predict(malicious_samples_gaussian))
acc_malicious_sp = accuracy_score([-1] * len(malicious_samples_sp), model.predict(malicious_samples_sp))

print(f"Accuracy on malicious samples with Gaussian noise: {acc_malicious_gaussian}")
print(f"Accuracy on malicious samples with Salt-and-Pepper noise: {acc_malicious_sp}")


# In[ ]:




