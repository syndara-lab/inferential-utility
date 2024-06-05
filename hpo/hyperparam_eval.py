"""Evaluation of the top hyperparameters (tuned for n=500) in n=50 and n=5000"""

import pandas as pd
import optuna
from optuna import load_study, create_study
import sys
sys.path.append("..")
from utils.disease import sample_disease
from utils.tuning import tuning_objective_multiple

if __name__ == "__main__":
    
    ### Load optuna study
    
    # Load study
    multiple_seeds = True # use different set of 5 seeds (used for model initialiation and train/val split) for each trial and pool results over these seeds
    model_class = 'CTGAN' # 'CTGAN' or 'TVAE'
    package = 'custom_sdv' # 'synthcity' to use Synthcity, 'custom_sdv' to use SDV (customized), 'tune_package' to set the package chosen as a tunable hyperparameter as well
    study_name_elements = [model_class, package, 'study']
    if multiple_seeds:
        study_name_elements.append('multiple') # add 'multiple' to study_name if multiple seeds are considered
    study_name = '_'.join(study_name_elements) # name of study
    study_storage = f'sqlite:///{study_name}.db'
    study = load_study(study_name=study_name, storage=study_storage) # load SQLite DB file to access the history of study
    
    # Create dataframe from study
    study_df = study.trials_dataframe()
    
    ### Performance with different sample size
    
    # Draw random sample from ground truth
    seed = 2023 
    data_n_50 = sample_disease(50, seed=seed)
    data_n_5000 = sample_disease(5000, seed=seed)
    discrete_columns = ['stage', 'therapy', 'death'] # columns with discrete values
    
    # Extract hyperparameters of top trials
    top = 3
    study_df_top = study_df.sort_values(by=['value'], ascending=False).head(n=top) # select top trials
    study_df_top_dict = study_df_top.to_dict('index') # transform to dictionary
    hyperparam_keys = [column for column in study_df.columns if 'params_' in column] # select hyperparameter columns
    
    # Create study with top hyperparameters
    study_n_50 = create_study()
    study_n_5000 = create_study()
    for index in study_df_top.index:
        hyperparams = {k[len('params_'):]: study_df_top_dict[index][k] for k in hyperparam_keys} # select hyperparameters subset of dictionary and remove 'params_' prefix
        study_n_50.enqueue_trial(hyperparams, user_attrs={'original_trial_number': index}) # add trial with these (fixed) hyperparameters and set original trial number as user attribute
        study_n_5000.enqueue_trial(hyperparams, user_attrs={'original_trial_number': index}) # add trial with these (fixed) hyperparameters and set original trial number as user attribute
    
    # Run study for n=50
    study_n_50.optimize(lambda trial: tuning_objective_multiple(trial, model_class=model_class, package=package, data=data_n_50, discrete_columns=discrete_columns),
                        n_trials=top)
    
    # Run study for n=5000
    study_n_5000.optimize(lambda trial: tuning_objective_multiple(trial, model_class=model_class, package=package, data=data_n_5000, discrete_columns=discrete_columns),
                          n_trials=top)