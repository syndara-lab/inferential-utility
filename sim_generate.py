"""Simulation study 1: sample original data and generate synthetic version(s) using different generative models."""

import numpy as np
import pandas as pd
import os
import warnings
from time import time
from utils.disease import sample_disease
from utils.custom_ctgan import CTGAN
from utils.custom_tvae import TVAE
from utils.custom_bayesian import BayesianNetworkDAGPlugin
from utils.custom_synthpop import synthpopPlugin
from optuna import load_study
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
import torch
import random
import sys

def generative_models(n_sample: int=200):
    """
    Specify generative models used in simulation study.
    """
    
    # Extract tuned hyperparameters for CTGAN
    study_ctgan = load_study(study_name='CTGAN_custom_sdv_study_multiple', storage=f'sqlite:///hpo/CTGAN_custom_sdv_study_multiple.db') # load study
    study_ctgan_df = study_ctgan.trials_dataframe()
    hyperparam_keys = [column for column in study_ctgan_df.columns if 'params_' in column] # select hyperparameter columns
    hyperparams_ctgan = study_ctgan_df.sort_values(by=['value'], ascending=False).loc[2, hyperparam_keys].to_dict() # select hyperparameters of third best trial
    hyperparams_ctgan = {k[len('params_'):]: hyperparams_ctgan[k] for k in hyperparams_ctgan.keys()} # remove 'params_' prefix
    hyperparams_ctgan['discriminator_lr'] = hyperparams_ctgan['generator_lr'] # set discriminator_lr equal to generator_lr
    hyperparams_ctgan['generator_dim'] = hyperparams_ctgan['generator_n_layers_hidden']*(hyperparams_ctgan['generator_n_units_hidden'],) # define generator layer dimensions
    hyperparams_ctgan['discriminator_dim'] = hyperparams_ctgan['discriminator_n_layers_hidden']*(hyperparams_ctgan['discriminator_n_units_hidden'],) # define discriminator layer dimensions
    hyperparams_ctgan['batch_size'] = min(n_sample, 200)
    for name in ['discriminator_n_layers_hidden', 'discriminator_n_units_hidden', 'generator_n_layers_hidden', 'generator_n_units_hidden']: # use the correct naming of hyperparameters in the CTGAN module
        del hyperparams_ctgan[name] 

    # Extract tuned hyperparameters for TVAE
    study_tvae = load_study(study_name='TVAE_custom_sdv_study_multiple', storage=f'sqlite:///hpo/TVAE_custom_sdv_study_multiple.db') # load study
    hyperparams_tvae = study_tvae.best_params # select hyperparameters of best trial
    hyperparams_tvae['compress_dims'] = hyperparams_tvae['n_layers']*(hyperparams_tvae['n_hidden_units'],) # define encoder layer dimensions
    hyperparams_tvae['decompress_dims'] = hyperparams_tvae['n_layers']*(hyperparams_tvae['n_hidden_units'],) # define decoder layer dimensions (here equal to encoder)
    hyperparams_tvae['batch_size'] = min(n_sample, 200)
    for name in ['n_layers', 'n_hidden_units']: # use the correct naming of hyperparameters in the TVAE module
        del hyperparams_tvae[name]
    
    # Define generative models (assign Synthcity Plugins() to synthcity_models object prior to calling this function)
    models = [synthcity_models.get('synthpop',
                                   continuous_columns=['age', 'biomarker'],
                                   categorical_columns=['therapy', 'death'],
                                   ordered_columns=['stage'],
                                   visit_sequence=[0,1,2,3,4], # age stage biomarker therapy death
                                   predictor_matrix=np.array([[0,0,0,0,0],
                                                              [1,0,0,0,0],
                                                              [0,1,0,0,0],
                                                              [0,0,0,0,0],
                                                              [1,1,0,1,0]])), # age stage biomarker therapy death
              synthcity_models.get('bayesian_network'),
              synthcity_models.get('bayesian_network_DAG',
                                   dag=[('age', 'stage'), ('stage', 'biomarker'), ('age', 'death'), ('stage', 'death'), ('therapy', 'death')]),
              synthcity_models.get('ctgan', batch_size=min(n_sample, 200)), # default hyperparameters
              CTGAN(**hyperparams_ctgan), # tuned hyperparameters
              synthcity_models.get('tvae', batch_size=min(n_sample, 200)), # default hyperparameters
              TVAE(**hyperparams_tvae), # tuned hyperparameters 
              synthcity_models.get('privbayes'), # default hyperparameters
              synthcity_models.get('dpgan', batch_size=min(n_sample, 200)), # default hyperparameters
              synthcity_models.get('pategan', batch_size=min(n_sample, 200))] # default hyperparameters
    
    return models

def train_model(original_data: pd.DataFrame, generative_model: any, discrete_columns: list[str]=None, seed: int=2024): 
    """
    Wrapper function for training generative model.
    """
    
    # Fix seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Use custom plugins (adapted from SDV)
    if 'custom' in generative_model.name():  
        
        with warnings.catch_warnings(): 
            # ignore warning on line 132 of rdt/transformers/base.py: 
            # FutureWarning: Future versions of RDT will not support the 'model_missing_values' parameter. 
            # Please switch to using the 'missing_value_generation' parameter to select your strategy.
            warnings.filterwarnings("ignore", lineno=132)
            generative_model.fit(original_data, discrete_columns=discrete_columns, seed=seed)
            
    # Use Synthcity plugins
    else:
        loader = GenericDataLoader(original_data, sensitive_features=list(original_data.columns))
        generative_model.fit(loader) # seed is fixed within Synthcity (random_state=0)
        
    return generative_model

def sample_synthetic(generative_model: any, m: int=1, seed: int=2024) -> pd.DataFrame:
    """
    Wrapper function for sampling synthetic data.
    """
    
    # Fix seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Use custom plugins (adapted from SDV)
    if 'custom' in generative_model.name():
        synthetic_data = generative_model.sample(n=m, seed=seed)
    
    # Use Synthcity plugins
    else:
        synthetic_data = generative_model.generate(count=m, 
                                                   seed=seed, # seed argument is used by synthpopPlugin
                                                   random_state=seed).dataframe() # random_state argument is used by other plugins
    
    return synthetic_data   

def simulation_study1(n_samples, start_run, n_runs, n_sets, sim_dir, discrete_columns):
    """
    Setup of simulation study.
    """
    
    # OUTER loop over number of observations per original data set
    for n_sample in n_samples: 
        
        # Define output folder to save files
        n_dir = f'n_{n_sample}/'
        if not os.path.exists(sim_dir + n_dir): 
            os.mkdir(sim_dir + n_dir) # create n_dir folder if it does not exist yet
            
        # Define generative models
        models = generative_models(n_sample) # batch_size hyperparameter depends on sample size

        # INNER loop over Monte Carlo runs
        for i in range(start_run, start_run+n_runs):
            
            # Print progress
            print(f'[n={n_sample}, run={i}] start')
            
            # Define output folder to save files
            out_dir = sim_dir + n_dir + f'run_{i}/'
            if not os.path.exists(out_dir): 
                os.mkdir(out_dir) # create out_dir folder if it does not exist yet
            
            # Simulate toy data
            original_data = sample_disease(n_sample, seed=i)
            original_data.to_csv(out_dir + 'original_data.csv', index=False) # export file
                                
            # Train generative models
            for model in models:
                
                try:
                    train_model(original_data, model, discrete_columns, seed=2024)                                   
                except Exception as e:
                    print(f'[n={n_sample}, run={i}] error with fitting {model.name()}: {e}')
            
            # Generate synthetic data
            for model in models:
                
                for j in range(n_sets):
                    
                    try:
                        synthetic_data = sample_synthetic(model, m=n_sample, seed=j) # generated synthetic data = size of original data
                        synthetic_data.to_csv(out_dir + model.name() + f'_{j}.csv', index=False) # export file
                    except Exception as e:
                        print(f'[n={n_sample}, run={i}, set={j}] error with generating from {model.name()}: {e}')
                                    
if __name__ == "__main__":
    
    # Presets 
    n_samples = [50,160,500,1600,5000] # number of observations per original dataset
    start_run = 0 # start index of Monte Carlo runs (0 when single job submission; int(sys.argv[1]) when parallel job submission)
    n_runs = 200 # number of Monte Carlo runs per number of observations
    n_sets = 1 # number of synthetic datasets generated per generative model
    
    sim_dir = 'simulation_study1/' # output of simulations
    if not os.path.exists(sim_dir): 
        os.mkdir(sim_dir) # create sim_dir folder if it does not exist yet
    
    discrete_columns = ['stage', 'therapy', 'death'] # columns with discrete values
    
    # Disable tqdm as default setting (used internally in Synthcity plugins)
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    
    # Load Synthcity plugins 
    synthcity_models = Plugins()
    synthcity_models.add('bayesian_network_DAG', BayesianNetworkDAGPlugin) # add the BayesianNetworkDAGPlugin to the collection
    synthcity_models.add('synthpop', synthpopPlugin) # add the synthpopPlugin to the collection

    # Run simulation study
    start_time = time()
    simulation_study1(n_samples, start_run, n_runs, n_sets, sim_dir, discrete_columns)
    print(f'Total run time: {(time()-start_time):.3f} seconds')