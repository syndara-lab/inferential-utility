"""Case study 1: calculate inferential utility metrics of original and synthetic datasets."""

import os
import pandas as pd
import numpy as np
import itertools
from itertools import product
from time import time
from utils.eval import avg_inv_KLdiv, common_rows_proportion, estimate_estimator, CI_coverage
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM

def load_data(n_runs: int, case_dir: str):
    """
    Load source files containing original and synthetic datasets.
    """ 
    
    # Create empty data structure for storage
    data = {}
    for i in range(n_runs):
        data[f'run_{i}'] = {}
    
    # Load data sets
    data_dir = [os.path.join(root, name) for root, dirs, files in os.walk(case_dir) for name in files]
    for dir in data_dir:
        
        if '.ipynb_checkpoint' in dir:
            continue # skip dot files
        
        dir_structure = dir[len(case_dir):].split('/') # remove case_dir prefix from name and split directory by '/'
        try: # load datasets per Monte Carlo run
            data[dir_structure[0]][dir_structure[1][:-4]] = pd.read_csv(dir, engine='pyarrow') # also remove '.csv' suffix from name
        except Exception as e:
            print(f'File {dir} not stored in \'data\' object because file has aberrant directory structure')
    
    # Remove empty simulations
    for i in range(n_runs):
        if not data[f'run_{i}']:
            del data[f'run_{i}']
    
    return data

def create_metadata(data: pd.DataFrame, export: bool=False, case_dir: str=None):
    """
    Create meta_data (or update if additional generative models have been added) per Monte Carlo run.
    """
    
    for run in data:
        
        all_data = data[run] # original and synthetic data sets
        meta_data = pd.DataFrame({'dataset_name': [name for name in all_data if name not in ['meta_data']],
                                  'n': [all_data[name].shape[0] for name in all_data if name not in ['meta_data']],
                                  'run': np.repeat(run, len([name for name in all_data if name not in ['meta_data']])),
                                  'generator': ['_'.join(name.split('_')[:-1]) for name in all_data if name not in ['meta_data']]})
        data[run]['meta_data'] = meta_data # store meta data per simulation
        
        if export:
            meta_data.to_csv(case_dir + n_sample + '/' + run + '/meta_data.csv') # export meta_data to .csv
    
    return data

def create_dummies(data: pd.DataFrame):
    """
    Create dummy variables from categorical variables.
    """
    
    for run in data:
        for dataset_name in [name for name in data[run] if name not in ['meta_data']]:
            dataset = data[run][dataset_name]
            dataset['income'] = pd.Categorical(dataset['income'], categories=['<=50K', '>50K']) # define all theoretical categories (so pd.get_dummies will create dummy for all theoretical categories)
            dataset_dummies = pd.get_dummies(dataset, columns=['income']) # create dummies only for 'income'
            data[run][dataset_name] = dataset.merge(dataset_dummies) # merge dummies
    
    return data

def calculate_estimates(data: pd.DataFrame):
    """
    Calculate estimates (store in meta data per Monte Carlo run).
    """
    
    for run in data:
        
        # univariate estimators
        for var in ['age']:
            for estimator in ['mean', 'mean_se']:
                data[run]['meta_data'][var + '_' + estimator] = estimate_estimator(
                    data=data[run], var=var, estimator=estimator)
        
        # multivariate estimators
        for var in [('income_>50K', 'age')]: # income is outcome
            tmp = np.array(estimate_estimator(data=data[run], var=var, estimator='logr')).transpose() # calculate regression coefficients
            columns = [var[0] + '_' + y + '_' + x for x, y in list(product(['logr', 'logr_se'], var[1:]))] # define names for regression coefficients
            for i in range(len(columns)):
                data[run]['meta_data'][columns[i]] = list(tmp[i]) # assign corresponding names and values
    
    return data

def combine_metadata(data: pd.DataFrame, export: bool=False, case_dir: str=None):
    """
    Combine all meta data over Monte Carlo runs.
    """
    
    data['meta_data'] = pd.DataFrame({})
    
    for run in data:
        if run=='meta_data': # if overall 'meta_data' was already created by calling combine_metadata
            continue
        else:
            data['meta_data'] = pd.concat([data['meta_data'],
                                           data[run]['meta_data']],
                                          ignore_index=True)
    
    if export:
        data['meta_data'].to_csv(case_dir + 'meta_data.csv') # export meta_data to .csv
    
    return data

def sanity_check(data: pd.DataFrame):
    """
    Calculate sanity checks.
    """
    
    data['meta_data']['sanity_common_rows_proportion'] = data['meta_data'].apply(
        lambda i: common_rows_proportion(data[i['run'].replace(' ', '_')]['original_data'],
                                         data[i['run'].replace(' ', '_')][i['dataset_name']]), axis=1)
    
    data['meta_data']['sanity_IKLD'] = data['meta_data'].apply(
        lambda i: avg_inv_KLdiv(data[i['run'].replace(' ', '_')]['original_data'],
                                data[i['run'].replace(' ', '_')][i['dataset_name']]), axis=1)
    
    return data

def estimates_check(meta_data: pd.DataFrame):
    """
    Check non-estimable estimates (too large or too small SE).
    """
    
    select_se = [column for column in meta_data.columns if column[-3:]=='_se'] # select columns with '_se' suffix
    
    for var_se in select_se:
        
        # very small SE: set estimate and SE to np.nan
        meta_data[var_se[:-len('_se')]].mask(meta_data[var_se] < 1e-10, np.nan, inplace=True) 
        meta_data[var_se].mask(meta_data[var_se] < 1e-10, np.nan, inplace=True)
            
        # very large SE: set estimate and SE to np.nan
        meta_data[var_se[:-len('_se')]].mask(meta_data[var_se] > 1e2, np.nan, inplace=True)
        meta_data[var_se].mask(meta_data[var_se] > 1e2, np.nan, inplace=True)
    
    return meta_data

def adult_groundtruth(case_dir: str):
    """
    Calculate ground truth from full Adult dataset.
    """
    
    # Load adult dataset
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status',
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                'hours_per_week', 'native_country', 'income'] 
    adult_gt = pd.concat([pd.read_csv(f'{case_dir}adult.data', sep=r'\s*,\s*', engine='python', names=features, na_values='?'),
                          pd.read_csv(f'{case_dir}adult.test', skiprows=1, sep=r'\s*,\s*', engine='python', names=features, na_values='?')]) 
    adult_gt.drop(['education_num'], axis=1, inplace=True) # drop redundant column
    adult_gt.dropna(axis=0, how='any', inplace=True) # only retain complete cases
    adult_gt['income'] = [income.replace('.', '') for income in adult_gt['income']] # remove '.' suffix from test labels
    adult_gt_dummies = pd.get_dummies(adult_gt, columns=['income']) # create dummies only for 'income'
    adult_gt = adult_gt.merge(adult_gt_dummies) # merge dummies
    
    # Population parameter (= estimates on full data set)
    logr_model = GLM(adult_gt['income_>50K'], sm.add_constant(adult_gt[['age']]), family=sm.families.Binomial(link=sm.families.links.Logit())).fit() # var[0] is outcome, var[1:] are predictors
    ground_truth = {'age_mean': np.mean(adult_gt['age']), 
                    'income_>50K_age_logr': logr_model.params['age']}
    
    return ground_truth

def inferential_utility(meta_data: pd.DataFrame, ground_truth):
    """
    Calculate inferential utility metrics.
    """
    
    # sample mean and logsitic regression coefficients
    for estimator in ['age_mean',
                      'income_>50K_age_logr']:
        
        # bias of population parameter
        meta_data[estimator + '_bias'] = meta_data.apply(
            lambda i: i[estimator] - ground_truth[estimator], axis=1) # population parameter
        
        # coverage of population parameter
        meta_data[estimator + '_coverage'] = meta_data.apply(
            lambda i:
            CI_coverage(estimate=i[estimator],
                        se=i[estimator + '_se'],
                        ground_truth=ground_truth[estimator], # population parameter
                        distribution='t' if 'mean' in estimator else 'standardnormal',
                        df=i['n']-1,
                        quantile=0.975), axis=1) # naive SE
        meta_data[estimator + '_coverage_corrected'] = meta_data.apply(
            lambda i:
            CI_coverage(estimate=i[estimator],
                        se=i[estimator + '_se'],
                        ground_truth=ground_truth[estimator], # population parameter
                        se_correct_factor=np.sqrt(2) if i['dataset_name']!='original_data' else 1, # corrected SE
                        distribution='t' if 'mean' in estimator else 'standardnormal',
                        df=i['n']-1,
                        quantile=0.975), axis=1)
    
    return meta_data

if __name__ == "__main__":

    # Presets
    n_runs = 200 # number of Monte Carlo runs per number of observations
    case_dir = 'case_study1/' # output of case study
    start_time = time()
    
    # Data preparation
    data = load_data(n_runs, case_dir) # load data
    data = create_metadata(data, export=False) # create (single) meta data per Monte Carlo run
    data = create_dummies(data) # create dummy variables from categorical variables (needed for estimating regression coefficients)
    
    # Calculate estimates per data set
    data = calculate_estimates(data)
    
    # Combine (overall) meta data over Monte Carlo runs 
    data = combine_metadata(data, export=False)
    
    # Calculate sanity checks
    data = sanity_check(data)
    
    # Only go further with meta_data (to reduce RAM memory)
    meta_data = data['meta_data']
    del data
    
    # Check non-estimable estimates (set to np.nan if too large or too small SE)
    meta_data = estimates_check(meta_data)
    
    # Calculate inferential utility metrics
    data_gt = adult_groundtruth(case_dir=case_dir)
    meta_data = inferential_utility(meta_data, ground_truth=data_gt)
    
    # Save to file
    meta_data.to_csv(case_dir + 'meta_data.csv') # export meta_data to .csv
    
    # Print run time
    print(f'Total run time: {(time()-start_time):.3f} seconds')