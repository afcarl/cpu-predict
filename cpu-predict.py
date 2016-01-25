#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import datetime
import getopt
import time

import numpy as np
import pandas as pd
from time import time
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as rmse
from sklearn.tree import DecisionTreeRegressor

from operator import itemgetter, attrgetter

import matplotlib.pylab as plt

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

_n_estimators = 100
_grid_search = False
_ensemble = False

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)#[:n_top]
    s = 'GridSearchScores-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"
    fd = open(s, "w")
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1), file = fd)
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)), file = fd)
        print("Parameters: {0}".format(score.parameters), file = fd)
        print("", file = fd)

def find_feature_importance(y,X, cuttoff=0):
    # This is important
    n = X.shape[0]
    features = pd.DataFrame(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.333 )


#    forest = GradientBoostingRegressor( n_estimators=_n_estimators,
#                                        max_features='sqrt',
#    )

    forest = RandomForestRegressor(n_estimators=_n_estimators, n_jobs=-1)
    if _grid_search is True:
#        '''
        param_grid = {"max_depth": [10,None,5],
                      "max_features": ['sqrt','log2',.1,.5,.9],
                      "min_samples_split": [1,5,9],
                      "min_samples_leaf": [1, 5, 10],
                      "bootstrap": [True],
                      "oob_score": [True,False],
        }

#        '''
        #param_grid = {"bootstrap": [True] }
        grid_search = GridSearchCV(forest, param_grid=param_grid, n_jobs=-1, verbose=100)
        start = time()
        grid_search.fit(X, y)
        report(grid_search.grid_scores_)

        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(grid_search.grid_scores_)))
        os.sys.exit()

    # learning_rate=1.0, max_depth=1, random_state=0,verbose=10)
    forest_fit = forest.fit(X_train, y_train)
    forest_predictions = forest_fit.predict(X)
    importances = forest_fit.feature_importances_
    std = np.std(forest.feature_importances_)
    indices = np.argsort(importances)[:-n-1:-1]

    #print("Feature ranking:")

    cuttoff_i = int(cuttoff)

    if cuttoff == 0:
        cuttoff_i = len(features)

    columns_to_use = ''
    
    #print ("Searching for %s columns" % cuttoff_i )
    for i,f in enumerate(indices):
        #print("%d. %s (%f) %i" % (f + 1, features[indices[f]], importances[indices[f]], indices[f]))
        if cuttoff_i <= 0:
            break
        #print("%d. %s (%f) %i" % (indices[i], features.columns[f], importances[f], f))

        columns_to_use += "%s|" % (features.columns[f])
        cuttoff_i = cuttoff_i - 1
    columns_to_use = columns_to_use[:-1]

    return columns_to_use,forest

def randForest(X,y,columns_to_use):
    X = X.filter(regex=columns_to_use)
    n = X.shape[0]
    features = pd.DataFrame(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.4 )

    forest = RandomForestRegressor(n_estimators=_n_estimators, n_jobs=-1)
    forest.fit(X_train, y_train)
    return forest

def gradBoostReg(X,y,columns_to_use):
    X = X.filter(regex=columns_to_use)
    n = X.shape[0]
    features = pd.DataFrame(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.4 )

    forest = GradientBoostingRegressor( n_estimators=_n_estimators)
    forest.fit(X_train, y_train)
    return forest

def decisReg(X,y,columns_to_use):
    X = X.filter(regex=columns_to_use)
    n = X.shape[0]
    features = pd.DataFrame(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.4 )

    forest = DecisionTreeRegressor()
    forest.fit(X_train, y_train)
    return forest


def ensemblePred(GR, RF, y,columns_to_use):

    df_test = pd.read_csv("test.csv")
    df_test_app = df_test.filter(regex=columns_to_use)
    scaler = StandardScaler()
    df_test_app = scaler.fit_transform(df_test_app)

    y_rf = RF.predict(df_test_app)
    y_gr = GR.predict(df_test_app)

    plt.plt(y_rf)
    plt.plt(y_gr)
    plt.show()
    df_test_sol = pd.DataFrame( (y_rf + y_gr)/2.  )

    writeResults(df_test_sol)


    
def trainAndFit(X,y,cuttoff=0):
    columns_to_use,model = find_feature_importance(y,X,cuttoff)

    X = X.filter(regex=columns_to_use)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.333 )
    model.fit(X_train,y_train)
    ypred = model.predict(X_test)
    print ("RMSE %s, cuttoff: %s, SCORE: %s" % ( np.sqrt(np.mean((y_test - ypred) ** 2)), cuttoff, model.score(X_test, y_test) ))
    print (columns_to_use)
    return columns_to_use,model

def writeResults(df_test_sol):
    df_test_sol.columns=['Prediction']

    df_test_sol[df_test_sol < 0] = 0

    df_test_sol.index += 1
    s = 'results-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
    print ("Writing %s" % s)
    df_test_sol.to_csv(s, index_label='Id')

def buildResults(columns_to_use,model):
    df_test = pd.read_csv("test.csv")
    df_test['hour'] = df_test['sample_time'].apply(lambda x: time.strptime(x, "%Y-%m-%d %H:%M:%S")[3] )
    df_test['weekday'] = df_test['sample_time'].apply(lambda x: time.struct_time(time.strptime(x, "%Y-%m-%d %H:%M:%S")).tm_wday )

    df_test_app = df_test.filter(regex=columns_to_use)

    df_test_sol = pd.DataFrame(model.predict(df_test_app))

    writeResults(df_test_sol)

def usage():
    print ("-c cuttoff")

if __name__=="__main__":

    cuttoff = 0

    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:e:gn", ["cuttoff=", "estimators=", "grid", "ensemble"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-c", "--cuttoff"):
            cuttoff = (a)
        elif o in ("-e", "--estimators"):
            _n_estimators = int(a)
        elif o in ("-g", "--grid"):
            _grid_search = True
        elif o in ("-n", "--ensemble"):
            _ensemble = True
        else:
            assert False, "unhandled option"

    df = pd.read_csv("train.csv")
    y = df['cpu_01_busy']

    X = df.drop('cpu_01_busy',1)
    X = X.drop('m_id',1)
    import time
    X['hour'] = df['sample_time'].apply(lambda x: time.strptime(x, "%Y-%m-%d %H:%M:%S")[3] )
    X['weekday'] = df['sample_time'].apply(lambda x: time.struct_time(time.strptime(x, "%Y-%m-%d %H:%M:%S")).tm_wday )

    X = X.drop('sample_time',1)

    columns_to_use,model = trainAndFit(X,y, cuttoff)

    if _ensemble is True:
        ensemblePred(gradBoostReg(X,y,columns_to_use),randForest(X,y,columns_to_use), y, columns_to_use)
    else:
        buildResults(columns_to_use,model)

