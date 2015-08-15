import time
import math
import random
from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import os, sys
import xgboost as xgb
import pandas as pd
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

def loadcsv():
    csvmap = {}
    csvmap['bill_of_materials'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/bill_of_materials.csv')
    csvmap['specs'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/specs.csv')
    csvmap['train_set'] =  pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/train_set.csv', parse_dates=[2,])
    csvmap['test_set'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/test_set.csv', parse_dates=[3,])
    csvmap['tube'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/tube.csv')
    csvmap['comp_adaptor'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_adaptor.csv')
    csvmap['comp_boss'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_boss.csv')
    csvmap['comp_elbow'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_elbow.csv')
    csvmap['comp_float'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_float.csv')
    csvmap['comp_hfl'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_hfl.csv')
    csvmap['comp_nut'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_nut.csv')
    csvmap['components'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/components.csv')
    csvmap['comp_other'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_other.csv')
    csvmap['comp_sleeve'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_sleeve.csv')
    csvmap['comp_straight'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_straight.csv')
    csvmap['comp_tee'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_tee.csv')
    csvmap['comp_threaded'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/comp_threaded.csv')
    csvmap['tube_end_form'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/tube_end_form.csv')
    csvmap['type_component'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/type_component.csv')
    csvmap['type_connection'] = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/type_connection.csv')
    csvmap['type_end_form']  = pd.read_csv('/home/aocampor/CaterpillarTubePricing/competition_data/type_end_form.csv')
    return csvmap

def MergeData(train_set, bill_of_materials, specs, tube):
    data = pd.merge(train_set, bill_of_materials, on='tube_assembly_id')
    data = pd.merge(data, specs, on='tube_assembly_id')
    data = pd.merge(data, tube, on='tube_assembly_id')
    return data
                
def FilterOnBracketPrice(data):
    data_bp = data[data['bracket_pricing'] == 1]
    data_nbp = data[data['bracket_pricing'] == 0]    
    return data_bp, data_nbp

def GetCostSplit(data_bp, data_nbp):
    cost_bp = data_bp['cost']
    cost_nbp = data_nbp['cost']
    return cost_bp, cost_nbp

def GetCost(data):
    #print data_nbp
    cost = data['cost']
    return cost

def ConvertToArraysSplit(data_bp, data_nbp, test_bp, test_nbp, cost_bp, cost_nbp):
    data_bp = np.array(data_bp)
    data_nbp = np.array(data_nbp)

    atest_bp = np.array(test_bp)
    atest_nbp = np.array(test_nbp)
    
    label_bp = np.array(cost_bp)
    label_nbp = np.array(cost_nbp)
    return data_bp, data_nbp, atest_bp, atest_nbp, label_bp, label_nbp

def ConvertToArrays(data, test, cost):
    data = np.array(data)
    atest = np.array(test)
    label = np.array(cost)
    return data, atest, label

def SettingsForXGBoost( eta = 0.2, gamma = 1, min_child_weight = 6, 
                        max_depth = 30, max_delta_step = 2):
    params = {}
    params["booster"] = "gbtree"
    #params["booster"] = "gblinear"
    params["objective"] = "reg:linear"
    params["bst:eta"] = eta
    params["bst:gamma"] = gamma
    #params["lambda"] = lambda1    
    params["bst:min_child_weight"] = min_child_weight
    #params["subsample"] = subsample
    #params["colsample_bytree"] = colsample_bytree
    #params["scale_pos_weight"] = scale_pos_weight
    params["silent"] = 1
    params["bst:max_depth"] = max_depth
    params["bst:max_delta_step"] = max_delta_step
    params["nthread"] = 16
    plst = list(params.items())
    return params, plst

def GettingMatricesSplit(data_bp, label_bp, atest_bp, data_nbp, label_nbp, atest_nbp):
    xgtrain_bp = xgb.DMatrix(data_bp, label=label_bp, missing = np.nan)
    xgtest_bp = xgb.DMatrix(atest_bp)
    xgcontr_bp = xgb.DMatrix(data_bp)

    xgtrain_nbp = xgb.DMatrix(data_nbp, label=label_nbp, missing = np.nan)
    xgtest_nbp = xgb.DMatrix(atest_nbp)
    xgcontr_nbp = xgb.DMatrix(data_nbp)
    return xgtrain_bp, xgtest_bp, xgcontr_bp, xgtrain_nbp, xgtest_nbp, xgcontr_nbp


def GettingMatrices(data, label, atest, data_test):
    xgtrain = xgb.DMatrix(data, label=label, missing = np.nan)
    xgtest = xgb.DMatrix(atest, missing = np.nan)
    xgcontr = xgb.DMatrix(data_test, missing = np.nan)
    return xgtrain, xgtest, xgcontr

def TrainSplit(plst, xgtrain_bp, xgtrain_nbp):
    num_rounds = 400
    model_bp = xgb.train(plst, xgtrain_bp, num_rounds)
    model_nbp = xgb.train(plst, xgtrain_nbp, num_rounds)
    return model_bp, model_nbp

def Train(plst, xgtrain):
    num_rounds = 3000
    model = xgb.train(plst, xgtrain, num_rounds)
    return model

def PredictSplit(xgtest_bp, xgtest_nbp, xgcontr_bp, xgcontr_nbp, model_bp, model_nbp):
    preds_bp = model_bp.predict(xgtest_bp)
    preds_nbp = model_nbp.predict(xgtest_nbp)

    control_bp = model_bp.predict(xgcontr_bp)
    control_nbp = model_bp.predict(xgcontr_nbp)
    
    return preds_bp , preds_nbp, control_bp, control_nbp

def Predict( xgtest,  xgcontr , model):
    preds = model.predict(xgtest)
    control = model.predict(xgcontr)
    return preds, control


def DoPlots(label, control):
    plt.scatter(label, control , s = label, c = control, marker="o")
    
    plt.xlabel('Real Cost', fontsize=18)
    plt.ylabel('Predicted cost', fontsize=16)
    ax = plt.gca()
    ax.set_axis_bgcolor('white')
    #plt.show()
    plt.savefig('scatterplot_bp.png')


def DoPlotsSplit(label_bp, control_bp , label_nbp, control_nbp):
    plt.scatter(label_bp, control_bp , s = 10, c = 'r', marker="o")
    plt.scatter(label_nbp, control_nbp , s = 10, c = 'b', marker="o")    
    
    plt.xlabel('Real Cost', fontsize=18)
    plt.ylabel('Predicted cost', fontsize=16)
    ax = plt.gca()
    ax.set_axis_bgcolor('white')
    plt.show()
    #plt.savefig('scatterplot_bp.png')

    
def CreatingCSVSplit(test_bp1, preds_bp, test_nbp1, preds_nbp):
    #print 'creating dataframes with outputs and merging them'
    preds_bp = pd.DataFrame({"id": test_bp1['id'], "cost": preds_bp})
    preds_nbp = pd.DataFrame({"id": test_nbp1['id'], "cost": preds_nbp})

    dfs = [preds_bp, preds_nbp]
    preds = pd.concat(dfs)
    preds['cost'] = preds['cost'].abs()
    #print 'creating csv file'
    preds.to_csv('benchmark.csv', index=False)

def CreatingCSV(test, preds):
    #print 'creating dataframes with outputs and merging them'
    #preds = pd.DataFrame({"id": test['id'], "cost": preds})
    preds = pd.DataFrame({"id": test, "cost": preds})    
    preds['cost'] = preds['cost'].abs()
    #print 'creating csv file'
    preds.to_csv('benchmark.csv', index=False)


if __name__ == "__main__":
    t0 = time.time()
    plt.style.use('ggplot')
    CSVMap = loadcsv()
    train_names = CSVMap['train_set'].columns.values.tolist()
    for key in CSVMap:
        if(key != 'train_set' and key != 'test_set'):
            names = CSVMap[key].columns.values.tolist()
            for nam in names:
                for namt in train_names:
                    if(nam == namt):
                        CSVMap['train_set'] = pd.merge(CSVMap['train_set'], CSVMap[key], on=nam)
                        CSVMap['test_set'] = pd.merge(CSVMap['test_set'], CSVMap[key], on=nam)

    CSVMap['train_set']['year'] = CSVMap['train_set'].quote_date.dt.year
    CSVMap['train_set']['month'] = CSVMap['train_set'].quote_date.dt.month
    CSVMap['train_set']['day'] = CSVMap['train_set'].quote_date.dt.day

    CSVMap['test_set']['year'] = CSVMap['test_set'].quote_date.dt.year
    CSVMap['test_set']['month'] = CSVMap['test_set'].quote_date.dt.month
    CSVMap['test_set']['day'] = CSVMap['test_set'].quote_date.dt.day    

    CSVMap['train_set'] = CSVMap['train_set'].drop(['quote_date'], axis = 1)
    CSVMap['test_set'] = CSVMap['test_set'].drop(['quote_date'], axis = 1)    

    train_names = CSVMap['train_set'].columns.values.tolist()
    test_names = CSVMap['test_set'].columns.values.tolist()

    CSVMap['train_set']['bracket_pricing'].replace('Yes','1', regex= True,  inplace=True)
    CSVMap['test_set']['bracket_pricing'].replace('Yes','1', regex= True,  inplace=True)
    CSVMap['train_set']['bracket_pricing'].replace('No','0', regex= True,  inplace=True)
    CSVMap['train_set']['bracket_pricing'] = CSVMap['train_set'][['bracket_pricing']].astype(float)
    CSVMap['test_set']['bracket_pricing'].replace('No','0', regex= True,  inplace=True)
    CSVMap['test_set']['bracket_pricing'] = CSVMap['test_set'][['bracket_pricing']].astype(float)                

    for item in train_names:
        lsplit = item.rsplit('end_')
        l1split = item.rsplit('_')
        if(len(lsplit) > 1 and len( l1split) > 2):
            CSVMap['train_set'][item].replace('Y','1', regex= True,  inplace=True)
            CSVMap['train_set'][item].replace('N','0', regex= True,  inplace=True)            
            CSVMap['train_set'][item] = CSVMap['train_set'][[item]].astype(float)
            CSVMap['test_set'][item].replace('Y','1', regex= True,  inplace=True)
            CSVMap['test_set'][item].replace('N','0', regex= True,  inplace=True)            
            CSVMap['test_set'][item] = CSVMap['test_set'][[item]].astype(float)            

            
    CSVMap['train_set'] = CSVMap['train_set'].drop(['tube_assembly_id'], axis = 1)
    CSVMap['test_set'] = CSVMap['test_set'].drop(['tube_assembly_id'], axis = 1)    

    CSVMap['train_set'] = CSVMap['train_set'].fillna(0)
    CSVMap['test_set'] = CSVMap['test_set'].fillna(0)
    
    CSVMap['train_set']['number_of_components'] = ( CSVMap['train_set']['quantity_1'] +
                                                    CSVMap['train_set']['quantity_2'] +
                                                    CSVMap['train_set']['quantity_3'] +
                                                    CSVMap['train_set']['quantity_4'] +
                                                    CSVMap['train_set']['quantity_5'] +
                                                    CSVMap['train_set']['quantity_6'] +
                                                    CSVMap['train_set']['quantity_7'] +
                                                    CSVMap['train_set']['quantity_8'] 
                                                    )
    CSVMap['test_set']['number_of_components'] = (  CSVMap['test_set']['quantity_1'] +
                                                    CSVMap['test_set']['quantity_2'] +
                                                    CSVMap['test_set']['quantity_3'] +
                                                    CSVMap['test_set']['quantity_4'] +
                                                    CSVMap['test_set']['quantity_5'] +
                                                    CSVMap['test_set']['quantity_6'] +
                                                    CSVMap['test_set']['quantity_7'] +
                                                    CSVMap['test_set']['quantity_8'] 
                                                    )


    train_names = [ 'supplier' , 'component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7', 'component_id_8',  'material_id', 'end_a', 'end_x', 'spec1', 'spec2', 'spec3', 'spec4', 'spec5', 'spec6', 'spec7', 'spec8', 'spec9', 'spec10', 'other'] 
        
    #for column in train_names:
    #    arra =  CSVMap['train_set'][column]
    #    mapa = {}
    #    for row in arra:
    #        if row in mapa:
    #            mapa[row] = mapa[row] + 1
    #        else:
    #            mapa[row] = 1
    #    largo = float(len(CSVMap['train_set'][column]))
    #    listemp = []
    #    listempe = []
    #    for key in mapa:
    #        listemp.append(key)
    #        listempe.append(float(mapa[key]*100/largo))
    #    temp = {column: pd.Series(listemp),
    #            column + '_freq': pd.Series(listempe)}
    #    temp1 = pd.DataFrame(temp)

        
    for column in train_names:
        arra =  CSVMap['test_set'][column]
        mapa = {}
        for row in arra:
            if row in mapa:
                mapa[row] = mapa[row] + 1
            else:
                mapa[row] = 1
        
        largo = len(CSVMap['test_set'][column]) 
        listemp = []
        listempe = []
        for key in mapa:
            listemp.append(key)
            listempe.append( float(mapa[key]*100/largo) )
        temp = {column: pd.Series(listemp),
                column + '_freq': pd.Series(listempe)}
        temp1 = pd.DataFrame(temp)
        CSVMap['test_set'] = pd.merge(CSVMap['test_set'], temp1, on = column)
        CSVMap['test_set'] = CSVMap['test_set'].drop([column], axis=1)
        CSVMap['train_set'] = pd.merge(CSVMap['train_set'], temp1, on = column)
        CSVMap['train_set'] = CSVMap['train_set'].drop([column], axis=1)


    #print 'getting cost'
    train_names = CSVMap['train_set'].columns.values.tolist()
    #print train_names
    for name in train_names:
    #    ##print name
        total = len(CSVMap['train_set'][name])
        valid = len(CSVMap['train_set'][CSVMap['train_set'][name] > 0])
        per = float(valid)*100./total
        if(per < 5):
            print per, name
            CSVMap['train_set'] = CSVMap['train_set'].drop([name], axis = 1)
            CSVMap['test_set'] = CSVMap['test_set'].drop([name], axis = 1)            


    #print CSVMap['train_set']['bracket_pricing']
    train_bp = CSVMap['train_set'][CSVMap['train_set']['bracket_pricing'] == 1 ]
    train_nbp = CSVMap['train_set'][CSVMap['train_set']['bracket_pricing'] == 0 ]

    test_bp = CSVMap['test_set'][CSVMap['test_set']['bracket_pricing'] == 1 ]
    test_nbp = CSVMap['test_set'][CSVMap['test_set']['bracket_pricing'] == 0 ]    
    #print train_nbp

    train_bp = train_bp.drop(['bracket_pricing'], axis=1)
    train_nbp = train_nbp.drop(['bracket_pricing'], axis=1)
    test_bp = test_bp.drop(['bracket_pricing'], axis=1)
    test_nbp = test_nbp.drop(['bracket_pricing'], axis=1)    
    
    idx_bp = test_bp['id']
    test_bp = test_bp.drop(['id'], axis = 1)    
    atest_bp = np.array(test_bp)
    xgtest_bp = xgb.DMatrix(atest_bp, missing = np.nan)    

    idx_nbp = test_nbp['id']
    test_nbp = test_nbp.drop(['id'], axis = 1)    
    atest_nbp = np.array(test_nbp)
    xgtest_nbp = xgb.DMatrix(atest_nbp, missing = np.nan)    

    #print idx_bp
    
    #idx = CSVMap['test_set']['id']
    #CSVMap['test_set'] = CSVMap['test_set'].drop(['id'], axis = 1)    
    #test = CSVMap['test_set']
    #atest = np.array(test)
    #xgtest = xgb.DMatrix(atest, missing = np.nan)    

    params, plst = SettingsForXGBoost( 0.8, 0.3, 5, 30, 8)

    num_rounds = 2000
    rounds = 10

    rows_bp = random.sample(train_bp.index, 15000)
    df_train_bp = train_bp.ix[rows_bp]
    df_test_bp = train_bp.drop(rows_bp)

    rows_nbp = random.sample(train_nbp.index, 1500)
    df_train_nbp = train_nbp.ix[rows_nbp]
    df_test_nbp = train_nbp.drop(rows_nbp)
     
    #rows = random.sample(CSVMap['train_set'].index, 15000)
    #df_train = CSVMap['train_set'].ix[rows]
    #df_test = CSVMap['train_set'].drop(rows)

    cost_bp = np.log(df_train_bp['cost']+1)
    cost_test_bp = np.log(df_test_bp['cost']+1)
    df_train_bp = df_train_bp.drop(['cost'], axis = 1)
    df_test_bp = df_test_bp.drop(['cost'], axis = 1)

    cost_nbp = np.log(df_train_nbp['cost']+1)
    cost_test_nbp = np.log(df_test_nbp['cost']+1)
    df_train_nbp = df_train_nbp.drop(['cost'], axis = 1)
    df_test_nbp = df_test_nbp.drop(['cost'], axis = 1)
    
    #cost = np.log(df_train['cost']+1)
    #cost_test = np.log(df_test['cost']+1)
    #df_train = df_train.drop(['cost'], axis = 1)
    #df_test = df_test.drop(['cost'], axis = 1)

    data_bp = df_train_bp
    data_bp = np.array(data_bp)
    data_test_bp = np.array(df_test_bp)
    label_bp = np.array(cost_bp)
    label_test_bp = np.array(cost_test_bp)

    data_nbp = df_train_nbp
    data_nbp = np.array(data_nbp)
    data_test_nbp = np.array(df_test_nbp)
    label_nbp = np.array(cost_nbp)
    label_test_nbp = np.array(cost_test_nbp)
    
    #data = df_train
    #data = np.array(data)
    #data_test = np.array(df_test)
    #label = np.array(cost)
    #label_test = np.array(cost_test)
    
    xgtrain_bp = xgb.DMatrix(data_bp, label=label_bp, missing = np.nan)        
    xgcontr_bp = xgb.DMatrix(data_test_bp, missing = np.nan)
    model_bp = xgb.train(plst, xgtrain_bp, num_rounds)

    xgtrain_nbp = xgb.DMatrix(data_nbp, label=label_nbp, missing = np.nan)        
    xgcontr_nbp = xgb.DMatrix(data_test_nbp, missing = np.nan)
    model_nbp = xgb.train(plst, xgtrain_nbp, num_rounds)

    #xgtrain = xgb.DMatrix(data, label=label, missing = np.nan)        
    #xgcontr = xgb.DMatrix(data_test, missing = np.nan)
    #model = xgb.train(plst, xgtrain, num_rounds)    
    
    preds_bp, control_bp = Predict(xgtest_bp, xgcontr_bp, model_bp)
    DoPlots(label_test_bp, control_bp )
    error_bp = control_bp - label_test_bp
    error_bp = error_bp**2
    errsum_bp = error_bp.sum()    

    preds_nbp, control_nbp = Predict(xgtest_nbp, xgcontr_nbp, model_nbp)
    DoPlots(label_test_nbp, control_nbp )
    error_nbp = control_nbp - label_test_nbp
    error_nbp = error_nbp**2
    errsum_nbp = error_nbp.sum()    

    #preds, control = Predict(xgtest, xgcontr, model)
    #DoPlots(label_test, control )
    #error = control - label_test
    #error = error**2
    #errsum = error.sum()    
    
    for i in range(1,rounds):
        print i
        rows_bp = random.sample(train_bp.index, 15000)
        df_train_bp = train_bp.ix[rows_bp]
        df_test_bp = train_bp.drop(rows_bp)

        rows_nbp = random.sample(train_nbp.index, 1500)
        df_train_nbp = train_nbp.ix[rows_nbp]
        df_test_nbp = train_nbp.drop(rows_nbp)
        
        #rows = random.sample(CSVMap['train_set'].index, 15000)
        #df_train = CSVMap['train_set'].ix[rows]
        #df_test = CSVMap['train_set'].drop(rows)

        cost_bp = np.log(df_train_bp['cost']+1)
        cost_test_bp = np.log(df_test_bp['cost']+1)
        df_train_bp = df_train_bp.drop(['cost'], axis = 1)
        df_test_bp = df_test_bp.drop(['cost'], axis = 1)
        
        cost_nbp = np.log(df_train_nbp['cost']+1)
        cost_test_nbp = np.log(df_test_nbp['cost']+1)
        df_train_nbp = df_train_nbp.drop(['cost'], axis = 1)
        df_test_nbp = df_test_nbp.drop(['cost'], axis = 1)
        
        #cost = np.log(df_train['cost']+1)
        #cost_test = np.log(df_test['cost']+1)
        #df_train = df_train.drop(['cost'], axis = 1)
        #df_test = df_test.drop(['cost'], axis = 1)
        
        data_bp = df_train_bp
        data_bp = np.array(data_bp)
        data_test_bp = np.array(df_test_bp)
        label_bp = np.array(cost_bp)
        label_test_bp = np.array(cost_test_bp)
        
        data_nbp = df_train_nbp
        data_nbp = np.array(data_nbp)
        data_test_nbp = np.array(df_test_nbp)
        label_nbp = np.array(cost_nbp)
        label_test_nbp = np.array(cost_test_nbp)
        
        #data = df_train
        #
        #data = np.array(data)
        #data_test = np.array(df_test)
        #label = np.array(cost)
        #label_test = np.array(cost_test)

        xgtrain_bp = xgb.DMatrix(data_bp, label=label_bp, missing = np.nan)        
        xgcontr_bp = xgb.DMatrix(data_test_bp, missing = np.nan)
        model_bp = xgb.train(plst, xgtrain_bp, num_rounds)
        
        xgtrain_nbp = xgb.DMatrix(data_nbp, label=label_nbp, missing = np.nan)        
        xgcontr_nbp = xgb.DMatrix(data_test_nbp, missing = np.nan)
        model_nbp = xgb.train(plst, xgtrain_nbp, num_rounds)

        #xgtrain = xgb.DMatrix(data, label=label, missing = np.nan)        
        #xgcontr = xgb.DMatrix(data_test, missing = np.nan)
        #model = xgb.train(plst, xgtrain, num_rounds)
        
        preds_bp1, control_bp1 = Predict(xgtest_bp, xgcontr_bp, model_bp)
        preds_nbp1, control_nbp1 = Predict(xgtest_nbp, xgcontr_nbp, model_nbp)        
        #preds1, control1 = Predict(xgtest, xgcontr, model)

        preds_bp = preds_bp + preds_bp1
        error_bp = control_bp - label_test_bp
        error_bp = error_bp**2
        errsum_bp = errsum_bp + error_bp.sum()
        
        preds_nbp = preds_nbp + preds_nbp1
        error_nbp = control_nbp - label_test_nbp
        error_nbp = error_nbp**2
        errsum_nbp = errsum_nbp + error_nbp.sum()
        
        #preds = preds + preds1
        #error = control - label_test
        #error = error**2
        #errsum = errsum + error.sum()

    preds_bp = preds_bp/rounds
    errsum_bp = errsum_bp / rounds
    preds_bp = np.exp(preds_bp) - 1
    print 'Error ' , math.sqrt(errsum_bp)

    preds_nbp = preds_nbp/rounds
    errsum_nbp = errsum_nbp / rounds
    preds_nbp = np.exp(preds_nbp) - 1
    print 'Error ' , math.sqrt(errsum_nbp)

    #preds = preds/rounds
    #errsum = errsum / rounds
    #preds = np.exp(preds) - 1
    #print 'Error ' , math.sqrt(errsum)

    preds_bp = pd.DataFrame({"id": idx_bp, "cost": preds_bp})
    preds_nbp = pd.DataFrame({"id": idx_nbp, "cost": preds_nbp})

    dfs = [preds_bp, preds_nbp]
    preds = pd.concat(dfs)
    preds['cost'] = preds['cost'].abs()
    #print 'creating csv file'
    preds.to_csv('benchmark.csv', index=False)
    
    #CreatingCSVSplit(idx_bp, preds_bp, idx_nbp, preds_nbp)
    
    #CreatingCSV(idx, preds)    
    
    t1 = time.time()
    print 'It took ' + str(t1 - t0) + 's'
    print 'Done!'
    
    
    

    
