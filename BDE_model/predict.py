import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
import rdkit.Chem as Chem
import descriptors
import Ipynb_importer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
import math
warnings.filterwarnings("ignore")

data = pd.read_excel('BDE_DATA.xlsx') 
data['Mols'] = data['SMILES'].apply(Chem.MolFromSmiles)
data['Mols'] = data['Mols'].apply(Chem.AddHs)
bond_types,x_SOB = descriptors.literal_bag_of_bonds(list(data['Mols']))
x_Estate = descriptors.truncated_Estate_fingerprints(list(data['Mols']))
x_cds = descriptors.custom_descriptor_set(list(data['Mols']))
SEC = np.concatenate((x_cds, x_SOB, x_Estate), axis=1)
CBD = pd.read_csv('CBD.csv',header = None) 
X = np.concatenate((CBD, SEC), axis=1)
Y = np.array(list(data['BDE']))

def normalize(X, a, b):
    return (X - a)/(b - a)

def denormalize(X, a, b):
    return X * (b - a) + a

def PADRE(X1,X2):
    if list(X1) != list(X2):
        X1X2 = X1-X2
        return np.concatenate([X1, X2, X1X2], axis=0)
    else:
        return 0
def PADRE_test(X1,X2):
    if list(X1) != list(X2):
        X1X2 = X1-X2
        return np.concatenate([X1, X2, X1X2], axis=0)
    else:
        return np.zeros((444))
    
MAE_train = []
RMSE_train =[]
r2_train = []

MAE_val = []
RMSE_val =[]
r2_val = []

MAE_test = []
RMSE_test =[]
r2_test = []

RS = ShuffleSplit(n_splits=20, test_size=0.2, random_state=2)
A = 1
for train_ind, test_ind in RS.split(X):
    X_train = X[train_ind]
    Y_train = Y[train_ind]
    X_test = X[test_ind]
    Y_test = Y[test_ind]
    
    padre_ = [PADRE(x,y) for x in X_train for y in X_train]
    bde_ = [x-y for x in Y_train for y in Y_train]

    h = 0
    padre = []
    bde_y = []
    for j in padre_:
        if j is not 0 :
            padre.append(j)
            bde_y.append(bde_[h])
        h = h + 1

    padre = np.array(padre)
    bde_y = np.array(bde_y)

    p_max = np.amax(padre); p_min = np.amin(padre)
    b_max = np.amax(bde_y); b_min = np.amin(bde_y)
    padre = normalize(padre,p_min,p_max)
    bde_y = normalize(bde_y,b_min,b_max)
    bde_mean = np.mean(Y_train)

    x_train, x_test, y_train, y_test = train_test_split(padre, bde_y, test_size=0.2, random_state=5)

    y_train_d = denormalize(y_train,b_min,b_max)
    y_test_d = denormalize(y_test,b_min,b_max)
   
    model = xgb.XGBRegressor(n_estimators=10, max_depth=9,learning_rate= 0.3,min_child_weight=5)
    model.fit(x_train,y_train)

    y_train_pred = denormalize(model.predict(x_train),b_min,b_max)
    y_test_pred = denormalize(model.predict(x_test),b_min,b_max)

    MAEerror_train = mean_absolute_error(y_train_d,y_train_pred)
    MAE_train.append(MAEerror_train)
    MAEerror_val = mean_absolute_error(y_test_d,y_test_pred)
    MAE_val.append(MAEerror_val)

    RMSEerror_train = math.sqrt(mean_squared_error(y_train_d,y_train_pred))
    RMSE_train.append(RMSEerror_train)
    RMSEerror_val = math.sqrt(mean_squared_error(y_test_d,y_test_pred))
    RMSE_val.append(RMSEerror_val)

    r2__train = r2_score(y_train_d,y_train_pred)
    r2_train.append(r2__train)
    r2__val = r2_score(y_test_d,y_test_pred)
    r2_val.append(r2__val)

    pre_mean = []
    model = xgb.XGBRegressor(n_estimators=10, max_depth=20,learning_rate= 0.3,min_child_weight=10)
    model.fit(padre, bde_y)
    
    for k in range(len(X_test)):  
        padre_test = normalize(np.array([PADRE_test(X_test[k],x) for x in X_train]),p_min,p_max)      
        predictions = denormalize(model.predict(padre_test),b_min,b_max) + Y_train
        mean = np.mean(predictions)
        pre_mean.append(mean)
    MAE = mean_absolute_error(Y_test, pre_mean)
    MAE_test.append(MAE)
    RMSE = math.sqrt(mean_squared_error(Y_test, pre_mean))
    RMSE_test.append(RMSE)
    r2 = r2_score(Y_test, pre_mean)
    r2_test.append(r2)
    
    A = A + 1
    
    print('r2:',r2)
    print('MAE:',MAE)
    print('RMSE:',RMSE)
    print('\n')
    
print('r2_test:',r2_test)
print('MAE_test:',MAE_test)
print('RMSE_test:',RMSE_test)