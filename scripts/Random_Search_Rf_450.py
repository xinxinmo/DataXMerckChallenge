# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Random Forest Model
from sklearn.ensemble import RandomForestRegressor

# Evaluation of the model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# Helper
import random
import csv
from timeit import default_timer as timer

#clean up output
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

global  ITERATION
ITERATION = 0
N_FOLDS = 5
MAX_EVALS = 1000
IN_FILE = '~/Merck/Data/ACT1_train_450.csv'
OUT_FILE = '../Results/RandomSearch_Tuning_450_3scores_iter1000.csv'

# Import data
train_1 = pd.read_csv(IN_FILE, dtype={"MOLECULE": object, "Act": float})

# Split training set and test set
y = train_1['Act'].values
train_1 = train_1.drop(['Act', 'Unnamed: 0'], axis = 1)
x = train_1.values
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size = 0.80, random_state = 0)

print(np.shape(Y_train), np.shape(X_train), np.shape(Y_test), np.shape(X_test))
Y_train = np.reshape(Y_train,(len(Y_train),1))
Y_test = np.reshape(Y_test,(len(Y_test),1))

# Define scoring
def r_square(y, y_pred):
    """ r^2 value defined by the competition host, r^2 = 1 indicates 100% prediction accuracy
    """
    avx = np.mean(y)
    avy = np.mean(y_pred)
    sum1, sumx, sumy = 0, 0, 0
    for i in range(len(y)):
        sum1 += (y[i] - avx)*(y_pred[i] - avy)
        sumx += (y[i] - avx)*(y[i] - avx)
        sumy += (y_pred[i] - avy)*(y_pred[i] - avy)
    return sum1*sum1/(sumx*sumy)

MAPE = []
def mean_ape(y_true, y_pred):
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

def mean_absolute_error(y_true,y_pred):
    return np.mean(np.abs((y_true-y_pred)))

# clean up y
Y_train = np.ravel(Y_train)

# R_square score
R_2 = make_scorer(r_square)

# Mean absolute percentage error
mean_ape = make_scorer(mean_ape)

# MAE
mae = make_scorer(mean_absolute_error)

# Baseline
clf_base = RandomForestRegressor(n_estimators = 500, n_jobs = -1)

# Perform n_fold cross validation
scores_base_r2 = cross_val_score(clf_base, X_train, Y_train, cv = N_FOLDS, scoring = R_2)
scores_base_mape = cross_val_score(clf_base, X_train, Y_train, cv = N_FOLDS, scoring = mean_ape)
scores_base_mae = cross_val_score(clf_base, X_train, Y_train, cv = N_FOLDS, scoring = mae)
print('5 fold CV R2 is %0.5f (+/- %0.5f)' %(scores_base_r2.mean(), scores_base_r2.std() * 2))
print('5 fold CV MAPE is %0.5f (+/- %0.5f)' %(scores_base_mape.mean(), scores_base_mape.std() * 2))
print('5 fold CV MAE is %0.5f (+/- %0.5f)' %(scores_base_mae.mean(), scores_base_mae.std() * 2))

param_grid = {
              "bootstrap": [True, False],
              "max_depth": list(np.linspace(1, 100, dtype=int)),
              "max_features": ['auto', 'log2'],
              "min_samples_split": list(np.linspace(2, 11, dtype=int)),
              "min_samples_leaf": list(np.linspace(2, 11, dtype=int)),
              "n_estimators": list(np.linspace(500, 1000, dtype=int))
             }


# Create a file and open a connection
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

# Write column names
headers = ['iteration', 'runtime', 'r2_score', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators', 'hyperparameters']
writer.writerow(headers)
of_connection.close()

def objective(hyperparameters):
    
    global ITERATION
    
    ITERATION += 1
        
    start = timer()
    clf = RandomForestRegressor(**hyperparameters, n_jobs = -1)
    # Perform n_fold cross validation
    r2_scores = cross_val_score(clf, X_train, Y_train, cv = N_FOLDS, scoring = R_2)
    #scores_mape = cross_val_score(clf, X_train, Y_train, cv = N_FOLDS, scoring = mean_ape)
    #scores_mae = cross_val_score(clf, X_train, Y_train, cv = N_FOLDS, scoring = mae)
    
    run_time = timer() - start
    
    # Extract the score
    score_r2 = r2_scores.mean()
    #score_mape = scores_mape.mean()
    #score_mae = scores_mae.mean()
    #print("iter:", ITERATION, "  r2 score:", score_r2, " mape score:", score_mape, " mae score:", score_mae)
    print("iter:", ITERATION, "  r2 score:", score_r2)
    
    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([ITERATION, run_time, score_r2, hyperparameters["max_depth"], hyperparameters["min_samples_split"], hyperparameters["min_samples_leaf"], hyperparameters["n_estimators"], hyperparameters])
    of_connection.close()
    
    return [score_r2 , hyperparameters, ITERATION]

def random_search(param_grid, max_evals = MAX_EVALS):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                  index = list(range(MAX_EVALS)))
    
    # Keep searching until reach max evaluations
    for i in range(MAX_EVALS):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

        # Evaluate randomly selected hyperparameters
        eval_results = objective(hyperparameters)
        
        results.loc[i, :] = eval_results
    
    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    
    return results 



random_results = random_search(param_grid)

print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')

import pprint
pprint.pprint(random_results.loc[0, 'params'])

# Get the best parameters
random_search_params = random_results.loc[0, 'params']

# Create, train, test model
clf_tuned = RandomForestRegressor(**random_search_params, n_jobs = -1)
# Perform n_fold cross validation
scores_tune_r2 = cross_val_score(clf_tuned, X_train, Y_train, cv = N_FOLDS, scoring = R_2)
scores_tune_mape = cross_val_score(clf_base, X_train, Y_train, cv = N_FOLDS, scoring = mean_ape)
scores_tune_mae = cross_val_score(clf_base, X_train, Y_train, cv = N_FOLDS, scoring = mae)
print('5 fold CV R2 is %0.5f (+/- %0.5f)' %(scores_tune_r2.mean(), scores_tune_r2.std() * 2))
print('5 fold CV MAPE is %0.5f (+/- %0.5f)' %(scores_tune_mape.mean(), scores_tune_mape.std() * 2))
print('5 fold CV MAE is %0.5f (+/- %0.5f)' %(scores_tune_mae.mean(), scores_tune_mae.std() * 2))
