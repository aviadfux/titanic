import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import stats

#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
#PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']


def substring_check(string, substring_list):

    if string is np.nan:
        return 'N'

    for substring in substring_list:
        if substring in string:
            return substring

    return 'N'

def extract_title(passenger):
    title = passenger['Title']

    if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 0
    elif title in ['Mrs', 'Countess', 'Mme']:
        return 1
    elif title in ['Miss', 'Mlle', 'Ms']:
        return 2
    elif title == 'Dr':
        if passenger['Sex'] == 'Male':
            return 0
        else:
            return 1
    else:
        return 3

def extract_title_(passenger_name):
    title = substring_check(passenger_name['Name'], title_list)

    if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 0
    elif title in ['Mrs', 'Countess', 'Mme']:
        return 1
    elif title in ['Miss', 'Mlle', 'Ms']:
        return 2
    elif title == 'Dr':
        if passenger_name['Sex'] == 'Male':
            return 0
        else:
            return 1
    else:
        return 3

def extract_deck(passenger):
    if passenger['Cabin'] is np.nan:
        return 'N'
    return substring_check(passenger['Cabin'], cabin_list)

def extract_deck_to_int(passenger_final_features):
    return passenger_final_features['Deck'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'N': 8}).astype(int)

def extract_sex(raw_data):
    return raw_data['Sex'].map({'male': 0, 'female': 1}).astype(int)

def extract_fare(raw_data):
    x = raw_data['Fare'].values.reshape(-1, 1) # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    series = pd.Series(x_scaled.reshape(-1))
    series.index += 1
    return series

def extract_embarked(raw_data):
    raw_data['Embarked'] = raw_data['Embarked'].fillna(value='S')
    return raw_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


def extract(raw_data, train_data = True):
    PassengerId = raw_data['PassengerId'].to_numpy().reshape(-1, 1)
    raw_data = raw_data.drop(['PassengerId'], axis=1)

    if train_data:
        survived = raw_data['Survived'].to_numpy().reshape(-1, 1)
        raw_data = raw_data.drop(['Survived'], axis=1)


    raw_data['Title'] = raw_data['Name'].map(lambda x: substring_check(x, title_list))
    raw_data['Title'] = raw_data.apply(extract_title, axis=1)
    raw_data = raw_data.drop(['Name'], axis=1)

    raw_data['Deck'] = raw_data['Cabin'].map(lambda x: substring_check(x, cabin_list))
    raw_data = raw_data.drop(['Cabin'], axis=1)

    raw_data['Embarked'] = extract_embarked(raw_data)
    raw_data['Sex'] = extract_sex(raw_data)
    raw_data['Deck'] = extract_deck_to_int(raw_data)

    label = LabelEncoder()
    raw_data['Age'] = raw_data['Age'].fillna(raw_data['Age'].median()).astype(float)
    raw_data['Fare'] = raw_data['Fare'].fillna(raw_data['Fare'].median()).astype(float)
    raw_data['AgeBin'] = label.fit_transform(pd.cut(raw_data['Age'].astype(int), 6))
    raw_data['FareBin'] = label.fit_transform(pd.qcut(raw_data['Fare'].astype(float), 5))

    raw_data = raw_data.drop(['Ticket', 'Age', 'Fare'], axis=1)

    raw_data['FamilySize'] = raw_data['SibSp'] + raw_data['Parch'] + 1

    raw_data['IsAlone'] = 1  # initialize to yes/1 is alone
    raw_data['IsAlone'].loc[raw_data['FamilySize'] > 1] = 0

    v = raw_data.drop(['Title'], axis=1)
    output = stats.zscore(v, axis=0)

    temp = stats.zscore(raw_data['Title'].to_numpy(), axis=0).reshape(-1, 1)
    output = np.concatenate((temp, output), axis=1)

    if (train_data):
        output = np.concatenate((output, survived), axis=1)
    else:
        output = np.concatenate((output, PassengerId), axis=1)

    return output