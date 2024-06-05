"""Case study 1: sample original data and generate synthetic version(s) from Adult Census Income dataset using CTGAN and Synthpop."""

import numpy as np
import pandas as pd
import os
from time import time
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from utils.custom_synthpop import synthpopPlugin
import torch
import random
import sys

def case_study1(n_sample, start_run, n_runs, n_sets, case_dir):
    """
    Setup of case study.
    """
    
    # Loop over Monte Carlo runs
    for i in range(start_run, start_run+n_runs):
        
        # Print progress
        print(f'[run={i}] start')
            
        # Define output folder to save files
        out_dir = case_dir + f'run_{i}/'
        if not os.path.exists(out_dir): 
            os.mkdir(out_dir) # create out_dir folder if it does not exist yet
            
        # Draw original data from full adult data set
        original_data = adult_gt.sample(n=n_sample, replace=False, random_state=i)
        original_data.to_csv(out_dir + 'original_data.csv', index=False) # export file
        
        # Fix seed for reproducibility
        random.seed(2024)
        np.random.seed(2024)
        torch.manual_seed(2024)
        
        # Define generative models
        models = [synthcity_models.get('ctgan'), # default hyperparameters
                  synthcity_models.get('synthpop',
                                       continuous_columns=['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week'],
                                       categorical_columns=['workclass', 'education', 'martial_status', 'occupation', 'relationship', 'race', 'sex',
                                                            'native_country', 'income'],
                                       ordered_columns=[],
                                       default_method=['normrank', 'logreg', 'cart', 'polr'])]
     
        # Train generative models
        for model in models:
            
            try:
                loader = GenericDataLoader(original_data, sensitive_features=list(original_data.columns))
                model.fit(loader) # seed is fixed within Synthcity (random_state=0)
            except Exception as e:
                print(f'[run={i}] error with fitting {model.name()}: {e}')
            
        # Generate synthetic data
        for model in models:
            
            for j in range(n_sets):              
                
                try:
                    synthetic_data = model.generate(count=n_sample, # generated synthetic data = size of original data
                                                    seed=j, # seed argument is used by synthpopPlugin
                                                    random_state=j).dataframe() # random_state argument is used by other plugins
                    synthetic_data.to_csv(out_dir + model.name() + f'_{j}.csv', index=False) # export file   
                except Exception as e:
                    print(f'[run={i}, set={j}] error with generating from {model.name()}: {e}')

if __name__ == "__main__":
        
    # Presets
    n_sample = 5000 # number of observations per original dataset
    start_run = 0 # start index of Monte Carlo runs (0 when single job submission; int(sys.argv[1]) when parallel job submission)
    n_runs = 200 # number of Monte Carlo runs per number of observations
    n_sets = 1 # number of synthetic data sets generated per model
    
    case_dir = 'case_study1/' # output of simulations
    if not os.path.exists(case_dir):
        os.mkdir(case_dir) # create case_dir folder if it does not exist yet
    
    # Disable tqdm as default setting (used internally in Synthcity's plugins)
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    
    # Load adult dataset
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status',
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                'hours_per_week', 'native_country', 'income'] 
    adult_gt = pd.concat([pd.read_csv(f'{case_dir}adult.data', sep=r'\s*,\s*', engine='python', names=features, na_values='?'),
                          pd.read_csv(f'{case_dir}adult.test', skiprows=1, sep=r'\s*,\s*', engine='python', names=features, na_values='?')]) 
    adult_gt.drop(['education_num'], axis=1, inplace=True) # drop redundant column
    adult_gt.dropna(axis=0, how='any', inplace=True) # only retain complete cases
    adult_gt['income'] = [income.replace('.', '') for income in adult_gt['income']] # remove '.' suffix from test labels
    
    # Load Synthcity plugins 
    synthcity_models = Plugins()
    synthcity_models.add('synthpop', synthpopPlugin) # add the synthpopPlugin to the collection
    
    # Run case study
    start_time = time()
    case_study1(n_sample, start_run, n_runs, n_sets, case_dir)
    print(f'Total run time: {(time()-start_time):.3f} seconds')