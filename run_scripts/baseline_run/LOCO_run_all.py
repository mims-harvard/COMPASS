#!/home/was966/micromamba/envs/responder/bin/python
#sbatch --mem 64G -c 12 -t 100:00:00 -p priority  ./ctct_run_all.py

import os
from tqdm import tqdm
from itertools import chain
import pandas as pd
import numpy as np
import random, torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale=1.3)
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


import sys
sys.path.insert(0, '/home/was966/Research/COMPASS/')
from baseline.immnue_score import immnue_score_methods
from compass.utils import plot_embed_with_label,plot_performance, score, score2

def onehot(S):
    assert type(S) == pd.Series, 'Input type should be pd.Series'
    dfd = pd.get_dummies(S, dummy_na=True)
    nanidx = dfd[dfd[np.nan].astype(bool)].index
    dfd.loc[nanidx, :] = np.nan
    dfd = dfd.drop(columns=[np.nan])*1.
    cols = dfd.sum().sort_values(ascending=False).index.tolist()
    dfd = dfd[cols]
    return dfd



import sys
sys.path.insert(0, '/home/was966/Research/mims-conceptor/')
from baseline.immnue_score import immnue_score_methods
from conceptor.utils import plot_embed_with_label,plot_performance, score

data_path = '../../../../paper/00_data/'
df_label = pd.read_pickle(os.path.join(data_path, 'ITRP.PATIENT.TABLE'))
df_tpm = pd.read_pickle(os.path.join(data_path, 'ITRP.TPM.TABLE'))
df_tpm.shape, df_label.shape


df_task = onehot(df_label.response_label)
size = df_label.groupby('cohort').size()
size = size.index + "\n(n = " + size.astype(str) + ")"
cohorts = df_label.groupby('cohort').size().sort_values().index.tolist()
cohorts


## for TIDE input
cohort_cancer_map = {'Allen': 'SKCM','Choueiri': 'KIRC', 'Gide': 'SKCM', 'Hugo': 'SKCM',
                     'IMVigor210': 'BLCA', 'IMmotion150': 'KIRC', 'Kim': 'STAD',
                     'Liu': 'SKCM', 'MGH': 'SKCM', 'Miao': 'KIRC', 'Riaz': 'SKCM', 
                     'Rose': 'BLCA', 'SU2CLC1': 'LUAD', 'SU2CLC2': 'LUSC',
                     'Snyder': 'BLCA', 'Zhao': 'GBM'}

## for NetBio input
cohort_target_map = {'Allen': 'CTLA4','Choueiri': 'PD1', 'Gide': 'PD1_CTLA4',
                     'Hugo': 'PD1', 'IMVigor210': 'PDL1', 'IMmotion150': 'PDL1',
                     'Kim': 'PD1', 'Liu': 'PD1', 'MGH':  'PD1_CTLA4',
                     'Miao': 'PD1_PDL1_CTLA4', 'Riaz': 'PD1', 'Rose': 'PD1_PDL1_CTLA4',
                     'SU2CLC1': 'PD1', 'SU2CLC2': 'PD1_PDL1_CTLA4',
                     'Snyder': 'PDL1', 'Zhao': 'PD1'}


def leave_one_cohort_out(cohorts):
    # Create a list of lists, each missing one element from the original list
    return [(cohorts[i], cohorts[:i] + cohorts[i+1:]) for i in range(len(cohorts))]
train_test_cohorts = leave_one_cohort_out(cohorts)



for mode in immnue_score_methods.keys():

    print('Evaludation on Model %s' % mode)
    
    work_dir = './LOCO/LOCO_%s' % (mode)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    res = []
    for test_cohort, train_cohorts in train_test_cohorts:

        train_cohort_name = 'Leave_%s_out' % test_cohort
        ## Get data for this cohort
        cohort_idx = df_label[df_label['cohort'].isin(train_cohorts)].index
        cohort_X = df_tpm.loc[cohort_idx]
        cohort_y = df_task.loc[cohort_idx]
        
        ## Get features for specific method, as of the training cohorts are mixed cancers, targets,
        ## We use the cancer type and drug target same as the test cohort
        cohort_target = cohort_target_map[test_cohort]
        cohort_cancer_type = cohort_cancer_map[test_cohort]
        
        Extractor = immnue_score_methods[mode]
        E = Extractor(cancer_type=cohort_cancer_type, drug_target=cohort_target)
        cohort_dfx = E(cohort_X)
        cohort_dfy = cohort_y['R']
    
        data_scaler = StandardScaler()
        train_X = data_scaler.fit_transform(cohort_dfx)
        train_y = cohort_dfy.values

        #print(train_X.shape)
        param_grid = {'penalty':['l2'], 'max_iter':[int(1e10)], 'solver':['lbfgs'],
                      'C':np.arange(0.1, 1, 0.1), 'class_weight':['balanced'] }
        model = LogisticRegression()
        
        gcv = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1).fit(train_X, train_y)
        best_C = gcv.best_params_['C']

        test_cohort_idx = df_label[df_label['cohort'] == test_cohort].index
        test_cohort_X = df_tpm.loc[test_cohort_idx]
        test_cohort_y = df_task.loc[test_cohort_idx]
        
        test_cohort_dfx = E(test_cohort_X)
        test_cohort_dfy = test_cohort_y['R']
        test_X = data_scaler.transform(test_cohort_dfx)
        
        pred_prob = gcv.best_estimator_.predict_proba(test_X)
        
        pred_testy = pd.DataFrame(pred_prob, index = test_cohort_dfy.index)
        pred_testy['train_cohort'] = train_cohort_name
        pred_testy['test_cohort'] = test_cohort    
        pred_testy['best_C'] = best_C
        pred_testy['mode'] = mode
        dfp = test_cohort_y.join(pred_testy)

        y_true, y_prob, y_pred = dfp['R'], dfp[1], dfp[[0, 1]].idxmax(axis=1)
        fig = plot_performance(y_true, y_prob, y_pred)
        fig.suptitle('cohort to cohort transfer: train: %s, test: %s' % (train_cohort_name, test_cohort), fontsize=16)
        fig.savefig(os.path.join(work_dir, 'CTCT_train_%s_test_%s.jpg' % (train_cohort_name, test_cohort)))
        res.append(dfp)

    dfs = pd.concat(res)
    dfp = dfs.groupby(['train_cohort', 'test_cohort']).apply(lambda x:score2(x['R'], x[1], x[[0, 1]].idxmax(axis=1)))
    mode_map = dfs.groupby('train_cohort')['mode'].unique().apply(lambda x:x[0])
    c_map = dfs.groupby('train_cohort')['best_C'].unique().apply(lambda x:x[0])
    
    #roc, prc, f1, acc, mcc
    dfp = dfp.apply(pd.Series)
    dfp.columns = ['ROC', 'PRC', 'F1', 'ACC', 'MCC']
    dfp = dfp.reset_index()
    dfp['mode'] = dfp.train_cohort.map(mode_map)
    dfp['best_C'] = dfp.train_cohort.map(c_map)
    
    dfs.to_csv(os.path.join(work_dir, 'source_performance.tsv'), sep='\t')
    dfp.to_csv(os.path.join(work_dir, 'metric_performance.tsv'), sep='\t')