"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""


import os
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def handle_categorical_feat(X_df):
    ''' It moves the categorical features to the last '''

    original_columns = []
    one_hot_columns = []
    for col_name, dtype in zip(X_df.dtypes.index, X_df.dtypes):
        if dtype == object:
            one_hot_columns.append(col_name)
        else:
            original_columns.append(col_name)
    
    X_df = X_df[original_columns + one_hot_columns]
    return X_df, one_hot_columns


def load_breast_data():
    breast = load_breast_cancer()
    feature_names = list(breast.feature_names)
    X, y = pd.DataFrame(breast.data, columns=feature_names), pd.Series(breast.target)
    dataset = {
        'problem': 'classification',
        'full': {
            'X': X,
            'y': y,
        },
        'd_name': 'breast',
        'search_lam': np.logspace(-1, 2.5, 15),
    }
    return dataset


def load_adult_data():
    # https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    df = pd.read_csv("./datasets/adult.data", header=None)
    df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols].copy()

    # X_df = pd.get_dummies(X_df)
    X_df, onehot_columns = handle_categorical_feat(X_df)

    y_df = df[label].copy()

    # Make it as 0 or 1
    y_df.loc[y_df == ' >50K'] = 1.
    y_df.loc[y_df == ' <=50K'] = 0.
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'adult',
        'search_lam': np.logspace(-2, 2, 15),
        'n_splines': 50,
        'onehot_columns': onehot_columns,
    }

    return dataset

def load_credit_data():
    # https://www.kaggle.com/mlg-ulb/creditcardfraud
    df = pd.read_csv(r'./datasets/creditcard.csv')
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]
    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'credit',
        'search_lam': np.logspace(-0.5, 2.5, 8),
    }

    return dataset

def load_churn_data():
    # https://www.kaggle.com/blastchar/telco-customer-churn/downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv/1
    df = pd.read_csv(r'./datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    train_cols = df.columns[1:-1] # First column is an ID
    label = df.columns[-1]

    X_df = df[train_cols].copy()
    # Handle special case of TotalCharges wronly assinged as object
    X_df['TotalCharges'][X_df['TotalCharges'] == ' '] = 0.
    X_df.loc[:, 'TotalCharges'] = pd.to_numeric(X_df['TotalCharges'])

    # X_df = pd.get_dummies(X_df)
    X_df, onehot_columns = handle_categorical_feat(X_df)

    y_df = df[label].copy() # 'Yes, No'

    # Make it as 0 or 1
    y_df[y_df == 'Yes'] = 1.
    y_df[y_df == 'No'] = 0.
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'churn',
        'search_lam': np.logspace(0, 3, 15),
        'onehot_columns': onehot_columns,
    }

    return dataset

def load_pneumonia_data(folder='/media/intdisk/medical/RaniHasPneumonia/'):
    featurename_file = os.path.join(folder, 'featureNames.txt')
    col_names = pd.read_csv(featurename_file, delimiter='\t', header=None, index_col=0).iloc[:, 0].values

    def read_data(file_path='pneumonia/RaniHasPneumonia/medis9847c.data'):
        df = pd.read_csv(file_path, delimiter='\t', header=None)
        df = df.iloc[:, :-1] # Remove the last empty wierd column
        df.columns = col_names
        return df

    df_train = read_data(os.path.join(folder, 'medis9847c.data'))
    df_test = read_data(os.path.join(folder, 'medis9847c.test'))

    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    X_df = df.iloc[:, :-1]
    y_df = df.iloc[:, -1]

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'test_size': 4352 / 14199,
        'd_name': 'pneumonia',
        'search_lam': np.logspace(0, 3, 15),
    }

    return dataset


def load_heart_data():
    # https://www.kaggle.com/sonumj/heart-disease-dataset-from-uci
    df = pd.read_csv('./datasets/HeartDisease.csv')
    label = df.columns[-2]
    train_cols = list(df.columns[1:-2]) + [df.columns[-1]]
    X_df = df[train_cols]
    y_df = df[label]

    # X_df = pd.get_dummies(X_df)
    X_df, onehot_columns = handle_categorical_feat(X_df)

    # Impute the missingness as 0
    X_df = X_df.apply(lambda col: col if col.dtype == object else col.fillna(0.), axis=0)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'heart',
        'search_lam': np.logspace(0, 3, 15),
        'onehot_columns': onehot_columns,
    }

    return dataset


def load_mimiciii_data():
    df_adult = pd.read_csv('./datasets/adult_icu.gz', compression='gzip')

    train_cols = [
        'age', 'first_hosp_stay', 'first_icu_stay', 'adult_icu', 'eth_asian',
        'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
        'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN',
        'admType_URGENT', 'heartrate_min', 'heartrate_max', 'heartrate_mean',
        'sysbp_min', 'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max',
        'diasbp_mean', 'meanbp_min', 'meanbp_max', 'meanbp_mean',
        'resprate_min', 'resprate_max', 'resprate_mean', 'tempc_min',
        'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 'spo2_mean',
        'glucose_min', 'glucose_max', 'glucose_mean', 'aniongap', 'albumin',
        'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
        'hematocrit', 'hemoglobin', 'lactate', 'magnesium', 'phosphate',
        'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc']

    label = 'mort_icu'

    X_df = df_adult[train_cols]
    y_df = df_adult[label]
    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'mimiciii',
        'search_lam': np.logspace(0, 3, 15),
    }

    return dataset


def load_mimicii_data():
    cols = ['Age', 'GCS', 'SBP', 'HR', 'Temperature',
        'PFratio', 'Renal', 'Urea',  'WBC', 'CO2', 'Na', 'K',
        'Bilirubin', 'AdmissionType', 'AIDS',
        'MetastaticCancer', 'Lymphoma', 'HospitalMortality']

    table = pd.read_csv('./datasets/mimic2.data', delimiter=' ', header=None)
    table.columns = cols

    X_df = table.iloc[:, :-1]
    y_df = table.iloc[:, -1]

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'mimicii',
        'search_lam': np.logspace(-2, 3.5, 15),
    }

    return dataset


def load_diabetes2_data(load_cache=False):
    cache_dataset_path = './datasets/diabetes_cache.pkl'

    if load_cache and os.path.exists(cache_dataset_path):
        print('Find the diabetes dataset. Load from cache.')
        with open(cache_dataset_path, 'rb') as fp:
            dataset = pickle.load(fp)
            return dataset

    df = pd.read_csv('./datasets/dataset_diabetes/diabetic_data.csv')

    x_cols = df.columns[2:-1]
    y_col = df.columns[-1]

    X_df = df[x_cols].copy()
    y_df = df[y_col].copy()

    y_df.loc[(y_df == 'NO') | (y_df == '>30')] = 0
    y_df.loc[y_df == '<30'] = 1
    # is_false = (y_df == 'NO')
    # y_df.loc[is_false] = 0
    # y_df.loc[~is_false] = 1
    y_df = y_df.astype(int)

    # Preprocess X
    X_df.loc[:, 'age'] = X_df.age.apply(lambda s: (int(s[1:s.index('-')]) + int(s[(s.index('-') + 1):-1])) / 2).astype(int)
    X_df.loc[:, 'weight'] = X_df.weight.apply(lambda s: 0. if s == '?' else ((float(s[1:s.index('-')]) + float(s[(s.index('-') + 1):-1])) / 2 if '-' in s else float(s[1:])))
    X_df.loc[:, 'admission_source_id'] = X_df.admission_source_id.astype('object')
    X_df.loc[:, 'admission_type_id'] = X_df.admission_type_id.astype('object')
    X_df.loc[:, 'discharge_disposition_id'] = X_df.discharge_disposition_id.astype('object')
    X_df.loc[:, 'change'] = X_df.change.apply(lambda s: 0 if s == 'No' else 1).astype(np.uint8)
    X_df.loc[:, 'diabetesMed'] = X_df.diabetesMed.apply(lambda s: 0 if s == 'No' else 1).astype(np.uint8)
    X_df.loc[:, 'metformin-pioglitazone'] = X_df['metformin-pioglitazone'].apply(lambda s: 0 if s == 'No' else 1).astype(np.uint8)
    X_df.loc[:, 'metformin-rosiglitazone'] = X_df['metformin-rosiglitazone'].apply(lambda s: 0 if s == 'No' else 1).astype(np.uint8)
    X_df.loc[:, 'glipizide-metformin'] = X_df['glipizide-metformin'].apply(lambda s: 0 if s == 'No' else 1).astype(np.uint8)
    X_df.loc[:, 'troglitazone'] = X_df['troglitazone'].apply(lambda s: 0 if s == 'No' else 1).astype(np.uint8)
    X_df.loc[:, 'tolbutamide'] = X_df['tolbutamide'].apply(lambda s: 0 if s == 'No' else 1).astype(np.uint8)
    X_df.loc[:, 'acetohexamide'] = X_df['acetohexamide'].apply(lambda s: 0 if s == 'No' else 1).astype(np.uint8)
    X_df = X_df.drop(['citoglipton', 'examide'], axis=1) # Only have NO in the data

    # diag_combined = X_df.apply(lambda x: set(
    #     [x.diag_1 for i in range(1) if x.diag_1 != '?'] + [x.diag_2 for i in range(1) if x.diag_2 != '?'] + [x.diag_3 for i in range(1) if x.diag_3 != '?']
    # ), axis=1)
    # diag_combined = diag_combined.apply(collections.Counter)

    # diag_multihot_encode = pd.DataFrame.from_records(diag_combined).fillna(value=0).astype(np.uint8)
    # diag_multihot_encode.columns = ['diag_%s' % str(c) for c in diag_multihot_encode.columns]

    X_df = X_df.drop(['diag_1', 'diag_2', 'diag_3'], axis=1)
    # X_df = pd.concat([X_df, diag_multihot_encode], axis=1)
    
    # X_df = pd.get_dummies(X_df)
    X_df, onehot_columns = handle_categorical_feat(X_df)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'diabetes2',
        'search_lam': np.logspace(-3, 2, 8),
        'n_splines': 50,
        'onehot_columns': onehot_columns,
    }

    with open(cache_dataset_path, 'wb') as op:
        pickle.dump(dataset, op)

    return dataset


def load_TCGA_data(test_split=0.33, n_splits=20, cosmic=True, random_state=1377, **kwargs):
    np.random.seed(random_state)

    filename = 'pancancer_cosmic.npz' if cosmic else 'pancancer_parsed.npz'
    x = np.load('datasets/TCGA/%s' % filename)['arr_0']

    # log transform
    x_df = pd.DataFrame(np.log10(x + 1))

    # append the column name
    transcript_names_path = 'transcript_names_cosmic' if cosmic else 'transcript_names'
    x_df.columns = np.load('datasets/TCGA/%s.npy' % transcript_names_path)

    # remove the columns with std as 0
    x_df = x_df.loc[:, (x.std(axis=0) > 0.)]

    covars = pd.read_csv('datasets/TCGA/potential_covariates.tsv', delimiter='\t')
    covars['label'] = np.logical_or(covars[['sample_type']] == 'Primary Blood Derived Cancer - Peripheral Blood',
        np.logical_or(covars[['sample_type']] == 'Additional Metastatic',
        np.logical_or(covars[['sample_type']] == 'Recurrent Tumor',
        np.logical_or(covars[['sample_type']] == 'Additional - New Primary',
        np.logical_or(covars[['sample_type']] == 'Metastatic',
                      covars[['sample_type']] == 'Primary Tumor')))))

    stratify_lookup = covars.groupby('submitter_id').label.apply(lambda x: len(x))
    covars['stratify'] = covars.submitter_id.apply(lambda x: stratify_lookup[x])
    covars = covars[['submitter_id', 'label', 'stratify']]
    covars['patient_idxes'] = list(range(covars.shape[0]))

    def group_shuffle_split():
        for _ in range(n_splits):
            train_lookups = []
            for num_record, df2 in covars.groupby('stratify'):
                train_lookup = df2.groupby('submitter_id').apply(lambda x: True)

                # randomly change them to be 0
                all_idxes = np.arange(len(train_lookup))
                np.random.shuffle(all_idxes)
                is_test_idxes = all_idxes[:int(len(train_lookup) * test_split)]

                train_lookup[is_test_idxes] = False
                train_lookups.append(train_lookup)
            train_lookups = pd.concat(train_lookups)

            covars['is_train'] = covars.submitter_id.apply(lambda x: train_lookups[x])

            train_idxes = covars.patient_idxes[covars.is_train].values
            test_idxes = covars.patient_idxes[~covars.is_train].values
            yield train_idxes, test_idxes

    y = covars['label'].astype(float)
    stratify = covars['stratify']

    dataset = {
        'problem': 'classification',
        'full': {
            'X': x_df,
            'y': y,
            'ss': group_shuffle_split(),
        },
        'd_name': 'TCGA-cosmic' if cosmic else 'TCGA-full',
    }

    return dataset

def load_support2cls2_data():
    # http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets
    df = pd.read_csv('./datasets/support2/support2.csv')

    one_hot_encode_cols = ['sex', 'dzclass', 'race' , 'ca', 'income']
    target_variables = ['hospdead']
    remove_features = ['death', 'slos', 'd.time', 'dzgroup', 'charges', 'totcst',
                       'totmcst', 'aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m',
                       'dnr', 'dnrday', 'avtisst', 'sfdm2']

    df = df.drop(remove_features, axis=1)

    rest_colmns = [c for c in df.columns if c not in (one_hot_encode_cols + target_variables)]
    # Impute the missing values for 0.
    df[rest_colmns] = df[rest_colmns].fillna(0.)

    # df = pd.get_dummies(df)
    df, onehot_columns = handle_categorical_feat(df)

    df['income'][df['income'].isna()] = 'NaN'
    df['income'][df['income'] == 'under $11k'] = ' <$11k'
    df['race'][df['race'].isna()] = 'NaN'

    X_df = df.drop(target_variables, axis=1)
    y_df = df[target_variables[0]]

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'support2cls2',
        'search_lam': np.logspace(1.5, 4, 15),
        'onehot_columns': onehot_columns,
    }

    return dataset


def load_onlinenewscls_data():
    dataset = load_onlinenews_data()

    y_df = dataset['full']['y']
    y_df[y_df < 1400] = 0
    y_df[y_df >= 1400] = 1

    X_df = dataset['full']['X']

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'onlinenewscls',
        'search_lam': np.logspace(0, 4, 15),
    }

    return dataset


def load_compass_data():
    df = pd.read_csv('./datasets/recid.csv', delimiter=',')

    target_variables = ['two_year_recid']

    # df = pd.get_dummies(df, prefix=one_hot_encode_cols)
    df, onehot_columns = handle_categorical_feat(df)

    X_df = df.drop(target_variables, axis=1)
    # X_df = X_df.astype('float64')
    y_df = df[target_variables[0]]

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'compass',
        'onehot_columns': onehot_columns,
    }

    return dataset


def load_compas2_data():
    df = pd.read_csv('./datasets/recid_score.csv', delimiter=',')

    target_variables = ['decile_score']

    # df = pd.get_dummies(df, prefix=one_hot_encode_cols)
    df, onehot_columns = handle_categorical_feat(df)

    X_df = df.drop(target_variables, axis=1)
    y_df = df[target_variables[0]]

    dataset = {
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'compas2',
        'onehot_columns': onehot_columns,
    }
    return dataset


def load_gcredit_data():
    ''' Load German Credit dataset '''
    df = pd.read_csv('./datasets/german_credit/credit.data', delimiter='\t', header=None)
    df.columns = [
        'checking_balance', 'months_loan_duration', 'credit_history', 'purpose', 'amount', 
        'savings_balance', 'employment_length', 'installment_rate', 'personal_status', 
        'other_debtors', 'residence_history', 'property', 'age', 'installment_plan', 'housing', 
        'existing_credits', 'dependents', 'telephone', 'foreign_worker', 'job', 'class']

    target_variables = ['class']

    # df, onehot_columns = handle_categorical_feat(df)

    X_df = df.drop(target_variables, axis=1)
    y_df = df[target_variables[0]]

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'gcredit',
    }
    return dataset


''' =================== Regression Datasets ==================== '''

def load_bikeshare_data():
    df = pd.read_csv('./datasets/bikeshare/hour.csv').set_index('instant')
    train_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                  'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    label = 'cnt'

    X_df = df[train_cols]
    y_df = df[label]
    dataset = {
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'bikeshare',
        'search_lam': np.logspace(-0.5, 2, 15),
    }

    return dataset


def load_calhousing_data():
    X, y = fetch_california_housing(data_home='./datasets/', download_if_missing=True, return_X_y=True)
    columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

    dataset = {
        'problem': 'regression',
        'full': {
            'X': pd.DataFrame(X, columns=columns),
            'y': pd.Series(y),
        },
        'd_name': 'bikeshare',
        'search_lam': np.logspace(-2, 1.5, 15),
        'discrete': False, # r-spline args, or it will fail
    }
    return dataset


def load_wine_data():
    df = pd.read_csv('./datasets/winequality-white.csv', delimiter=';')
    train_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    label = 'quality'

    X_df = df[train_cols]
    y_df = df[label]
    dataset = {
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'wine',
        'search_lam': np.logspace(0, 3.5, 15),
    }

    return dataset


def load_support2reg2_data():
    df = pd.read_csv('./datasets/support2/support2.csv')

    one_hot_encode_cols = ['sex', 'dzclass', 'race' , 'ca', 'income']
    target_variables = ['slos']
    remove_features = ['death', 'd.time', 'dzgroup', 'charges', 'totcst',
                       'totmcst', 'aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m',
                       'dnr', 'dnrday', 'avtisst', 'sfdm2', 'hospdead']

    df = df.drop(remove_features, axis=1)

    rest_colmns = [c for c in df.columns if c not in (one_hot_encode_cols + target_variables)]
    # Impute the missing values for 0.
    df[rest_colmns] = df[rest_colmns].fillna(0.)

    # df = pd.get_dummies(df)
    df, onehot_columns = handle_categorical_feat(df)

    df['income'][df['income'].isna()] = 'NaN'
    df['income'][df['income'] == 'under $11k'] = ' <$11k'
    df['race'][df['race'].isna()] = 'NaN'

    X_df = df.drop(target_variables, axis=1)
    # X_df = X_df.astype('float64')
    y_df = df[target_variables[0]]

    dataset = {
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'support2reg2',
        'search_lam': np.logspace(0.5, 4, 15),
        'onehot_columns': onehot_columns,
    }

    return dataset


def load_onlinenews_data():
    df = pd.read_csv('./datasets/onlinenews/OnlineNewsPopularity.csv')

    label = ' shares'
    train_cols = [' timedelta', ' n_tokens_title', ' n_tokens_content',
       ' n_unique_tokens', ' n_non_stop_words', ' n_non_stop_unique_tokens',
       ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos',
       ' average_token_length', ' num_keywords', ' data_channel_is_lifestyle',
       ' data_channel_is_entertainment', ' data_channel_is_bus',
       ' data_channel_is_socmed', ' data_channel_is_tech',
       ' data_channel_is_world', ' kw_min_min', ' kw_max_min', ' kw_avg_min',
       ' kw_min_max', ' kw_max_max', ' kw_avg_max', ' kw_min_avg',
       ' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares',
       ' self_reference_max_shares', ' self_reference_avg_sharess',
       ' weekday_is_monday', ' weekday_is_tuesday', ' weekday_is_wednesday',
       ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday',
       ' weekday_is_sunday', ' is_weekend', ' LDA_00', ' LDA_01', ' LDA_02',
       ' LDA_03', ' LDA_04', ' global_subjectivity',
       ' global_sentiment_polarity', ' global_rate_positive_words',
       ' global_rate_negative_words', ' rate_positive_words',
       ' rate_negative_words', ' avg_positive_polarity',
       ' min_positive_polarity', ' max_positive_polarity',
       ' avg_negative_polarity', ' min_negative_polarity',
       ' max_negative_polarity', ' title_subjectivity',
       ' title_sentiment_polarity', ' abs_title_subjectivity',
       ' abs_title_sentiment_polarity', ' shares']

    X_df = df[train_cols]
    y_df = df[label]

    # Capped the largest shares to 5000 to avoid extreme values
    y_df[y_df > 5000] = 5000

    dataset = {
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
        'd_name': 'onlinenews',
        'search_lam': np.logspace(-3, 2, 12),
        'discrete': False, # r-spline args, or it will fail
    }

    return dataset


def load_semi_synthetic_dataset(d_name, cache_dir='./datasets/ss_datasets/', overwrite=False):
    ''' 
    E.g. "ss_pneumonia_b0_r1377_xgb-d1_sh0.8_inc0.5" 
    
    '''
    import models_utils

    tmp = d_name.split('_')
    real_d_name = tmp[1]
    binarize = int(tmp[2][1:])
    random_state = int(tmp[3][1:])
    the_rest = tmp[4:]

    if binarize:
        non_b_d_name = 'ss_%s_b0_r%d_%s' % (real_d_name, random_state, '_'.join(the_rest))
        d_set = load_semi_synthetic_dataset(non_b_d_name, cache_dir=cache_dir, overwrite=overwrite)
        assert d_set['problem'] == 'regression', 'Only log-odds can be binarized!'

        r = np.random.RandomState(random_state)
        targets = r.binomial(1, 1. / (1. + np.exp(-d_set['full']['y'].values)))
        d_set['full']['y'] = pd.Series(targets)
        d_set['problem'] = 'classification'
        d_set['d_name'] = d_name

        print('Finish loading dataset %s' % d_name)
        return d_set

    if tmp[-1].startswith('inc'):
        # Load original dataset
        d_name_without_inc = '_'.join(tmp[:-1])
        dataset = load_semi_synthetic_dataset(d_name_without_inc, cache_dir=cache_dir, overwrite=overwrite)

        # Do the increment part
        increment_fold = float(tmp[-1][3:])
        increment_X = get_shuffle_X(dataset['full']['X'], increment_fold, random_state=random_state)

        # Produce target by model
        models = pickle.load(open(dataset['models_path'], 'rb'))
        increment_y = gen_targets(increment_X, models)

        dataset['full']['X'] = pd.concat([dataset['full']['X'], increment_X], axis=0).reset_index(drop=True)
        dataset['full']['y'] = pd.concat([dataset['full']['y'], increment_y], axis=0).reset_index(drop=True)
        dataset['inc'] = increment_fold
        return dataset
    
    if tmp[-1].startswith('sh'):
        # Load original dataset
        d_name_without_inc = '_'.join(tmp[:-1])
        dataset = load_semi_synthetic_dataset(d_name_without_inc, cache_dir=cache_dir, overwrite=overwrite)

        models = pickle.load(open(dataset['models_path'], 'rb'))

        shuffle_ratio = 1. if len(tmp[-1]) == 2 else float(tmp[-1][2:])
        print('Shuffle the dataset. The ratio is %.2f' % shuffle_ratio)

        r = np.random.RandomState(random_state)
        X_shuffled = np.apply_along_axis(r.permutation, axis=0, arr=dataset['full']['X'])
        if shuffle_ratio < 1.:
            N = X_shuffled.shape[0]
            X_shuffled = np.concatenate([
                dataset['full']['X'].values[:int(N * (1. - shuffle_ratio))], 
                X_shuffled[:int(N * shuffle_ratio)],
            ], axis=0)

        dataset['full']['X'] = pd.DataFrame(X_shuffled, columns=dataset['full']['X'].columns)
        dataset['full']['y'] = gen_targets(dataset['full']['X'], models)
        return dataset
    
    filepath = os.path.join(cache_dir, '%s.pkl' % d_name)
    if not overwrite and os.path.exists(filepath):
        print('Find cache. Finish loading dataset %s' % d_name)
        d_set = pickle.load(open(filepath, 'rb'))

        # Make X as float64 to fix the rspline problem
        # d_set['full']['X'] = d_set['full']['X'].astype('float64')
        return d_set

    d_set = load_data(real_d_name)
    X_train, y_train, problem = d_set['full']['X'], d_set['full']['y'], d_set['problem']

    # Reuse the model if exists
    models_path = os.path.join(cache_dir, '%s_model.pkl' % d_name)
    if os.path.exists(models_path):
        from models_utils import mypickle_load
        models = mypickle_load(models_path)
    else:
        models = []
        for model_name in the_rest:
            model = models_utils.get_model(X_train, y_train, problem, model_name, random_state)
            models.append(model)

    targets = gen_targets(X_train, models)

    if problem == 'classification':
        d_set['problem'] = 'regression'

    # Make the output as (N,) with df format
    d_set['full']['y'] = targets
    d_set['d_name'] = d_name

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    
    if not os.path.exists(models_path):
        pickle.dump(models, open(models_path, 'wb'))

    d_set['models_path'] = models_path

    pickle.dump(d_set, open(filepath, 'wb'))
    return d_set


def gen_targets(X_df, models):
    targets = []
    for model in models:
        if hasattr(model, 'predict_proba'):
            X_prob = model.predict_proba(X_df)[:, 1]
            target = np.log(X_prob) - np.log(1. - X_prob)
        else:
            target = model.predict(X_df)

        targets.append(target)

    # Take average voting
    targets = np.mean(targets, axis=0)
    return pd.Series(targets)


def get_shuffle_X(X_df, fold, random_state=1377):
    ''' Generate new X by shuffling the columns. Increase this number of fold '''
    X_df = X_df.reset_index(drop=True)

    r = np.random.RandomState(random_state)
    all_Xs = []

    total_upper_folds = int(np.ceil(fold))
    for _ in range(total_upper_folds):
        new_X_df = X_df.copy()

        for idx, view in enumerate(np.rollaxis(new_X_df.values, 1)):
            r.shuffle(view)
            new_X_df.iloc[:, idx] = view

        all_Xs.append(new_X_df)

    all_Xs = pd.concat(all_Xs, axis=0).reset_index(drop=True)
    all_Xs = all_Xs.iloc[:int(fold * X_df.shape[0])]
    return all_Xs


def load_data(d_name):
    ''' Entry point '''
    if d_name.startswith('ss'):
        return load_semi_synthetic_dataset(d_name)

    load_fn = eval('load_%s_data' % d_name)
    return load_fn()


def load_train_test_data(dataset, split_idx, n_splits, test_size, random_state=1377):
    split_cls = StratifiedShuffleSplit if dataset['problem'] == 'classification' else ShuffleSplit
    train_test_ss = split_cls(n_splits=n_splits, test_size=test_size, random_state=random_state)
    idxes_generator = train_test_ss.split(dataset['full']['X'], dataset['full']['y'])

    for the_split_idx, (train_idx, test_idx) in enumerate(idxes_generator):
        if the_split_idx == split_idx:
            break

    X_train, y_train = dataset['full']['X'].iloc[train_idx], dataset['full']['y'].iloc[train_idx]
    X_test, y_test = dataset['full']['X'].iloc[test_idx], dataset['full']['y'].iloc[test_idx]
    return X_train, X_test, y_train, y_test
