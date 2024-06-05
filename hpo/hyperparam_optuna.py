"""Hyperparameter tuning of CTGAN and TVAE using Optuna."""

import optuna
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import sys
sys.path.append("..")
from utils.disease import sample_disease
from utils.tuning import tuning_objective_single, tuning_objective_multiple

if __name__ == "__main__":
    
    ### Presets
    
    # Seed 
    seed = 2023
    
    # Draw random sample from ground truth
    n_samples = 500 # sample size
    disease_data = sample_disease(n_samples, seed=seed)
    discrete_columns = ['stage', 'therapy', 'death'] # columns with discrete values
    
    # Generative model to tune
    cv_folds = 5 # number of folds used for k-fold cross-validated study objective score
    model_class = 'CTGAN' # 'CTGAN' or 'TVAE'
    package = 'custom_sdv' # 'synthcity' to use Synthcity, 'custom_sdv' to use SDV (customized), 'tune_package' to set the package chosen as a tunable hyperparameter as well
    
    # Study
    multiple_seeds = True # use different set of 5 seeds (used for model initialization and train/val split) for each trial and pool results over these seeds
    study_n_trials = None # total number of trials, set to None if timeout is used
    study_n_startup_trials = 100 # number of random startup trials
    study_timeout = 12*60*60 # stop study after the given number of seconds, set to None if n_trials is used
    study_direction = 'maximize' # maximize the study objective
    
    study_sampler = TPESampler(seed=seed, n_startup_trials=study_n_startup_trials) # random sampling is used instead of the TPE algorithm until the given number of n_startup_trials finish in the same study
    study_pruner = MedianPruner(n_startup_trials=10) # prune if the trial’s best intermediate result is worse than median of intermediate results of previous trials at the same step
    study_n_jobs = 1 # number of parallel trials
    study_name_elements = [model_class, package, 'study']
    if multiple_seeds:
        study_name_elements.append('multiple') # add 'multiple' to study_name if multiple seeds are considered
    study_name = '_'.join(study_name_elements) # name of study
    study_storage = f'sqlite:///{study_name}.db' # save study as SQLite DB file to access the history of study (and also enable multi-node optimization)
    study_objective = tuning_objective_multiple if multiple_seeds else tuning_objective_single # scores are default normalized if multiple seeds are considered (to make scores scale-independent from dataset size) - if not desired then set argument normalize_score=False
    
    ### Optimization
    
    # Disable tqdm as default setting (used internally in Synthcity plugins)
    if package!='custom_sdv':
        from tqdm import tqdm
        from functools import partialmethod
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    
    # Tree-structured Parzen estimators (TPE) optimization algorithm
    study = create_study(study_name=study_name, storage=study_storage, direction=study_direction, sampler=study_sampler, pruner=study_pruner) # load_if_exists=True to resume existing study 
    study.optimize(lambda trial: study_objective(trial, model_class=model_class, package=package, data=disease_data, cv_folds=cv_folds, seed=seed, discrete_columns=discrete_columns), 
                   n_trials=study_n_trials, timeout=study_timeout, n_jobs=study_n_jobs)