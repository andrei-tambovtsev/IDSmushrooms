import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def apriori_apply(data):
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    freq_itemsets = apriori(data, min_support=0.25, use_colnames=True)
    ar = association_rules(freq_itemsets, metric='confidence', min_threshold=0.5)
    ar.sort_values('lift', inplace=True, ascending=False)
    print(1)


if __name__ == '__main__':
    # Loading data
    data = pd.read_csv('mushrooms.csv')
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.min_rows', 100)

    # One-hot encoding data
    parts_hot_encoded = []
    hot_encoded = pd.DataFrame()
    for c in data.columns:
        parts_hot_encoded += [pd.get_dummies(data[c], prefix=c)]
    data = pd.concat(parts_hot_encoded, axis=1)

    # Dropping unnecessary columns
    # Veil type only has one value
    data.drop(['class_p', 'bruises_f', 'gill-attachment_a', 'gill-size_n', 'stalk-shape_t', 'veil-type_p'],
              axis=1, inplace=True)

    #apriori_apply(data)

    data_edibility = data['class_e']
    data.drop(['class_e'], axis=1, inplace=True)

    # Drop not-so-important features
    for c in data.columns:
        if 'odor' not in c and 'stalk-root' not in c and 'spore-print-color' not in c:
            data.drop([c], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(data, data_edibility, random_state=0)

    # See how well the models do
    knn = KNeighborsClassifier(n_neighbors=5, leaf_size=1000, metric='minkowski', p=1)
    knn.fit(X_train, y_train)
    print(knn.score(X_test, y_test))

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))