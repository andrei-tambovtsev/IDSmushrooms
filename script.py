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


# Was useful for viewing some files
from sklearn.tree import DecisionTreeClassifier


def apriori_apply(data):
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    freq_itemsets = apriori(data, min_support=0.25, use_colnames=True)
    ar = association_rules(freq_itemsets, metric='confidence', min_threshold=0.5)
    ar.sort_values('lift', inplace=True, ascending=False)
    print(ar)


def distinct_rows(data):
    # Lets see how many distinct rows we actually have
    distinct = dict()
    for i, row in data.iterrows():
        s = str(row.array.astype(int))
        if s not in distinct:
            distinct[s] = 0
        distinct[s] += 1
    # No duplicates were found at all! This is strange


def create_decision_tree(data_X, data_y):
    # Decision tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    import graphviz

    dt = DecisionTreeClassifier(random_state=0, criterion='gini')
    dt = dt.fit(data_X, data_y)
    dot_data = tree.export_graphviz(dt, out_file=None,
                                    feature_names=data_X.columns,
                                    class_names=True,
                                    filled=True, rounded=True,
                                    special_characters=False)
    graph = graphviz.Source(dot_data)
    graph.render(comment,view=True)  # Creates a file


def trim_columns(data_X, keep):
    # Drop not-so-important features (those that match keep are important)
    # Found those using Ridge/Lasso regression coeficients
    ans = data_X.copy()

    for c in data_X.columns:
        to_drop = True
        for k in keep:
            if k in c:
                to_drop = False
        if to_drop:
            ans.drop([c], axis=1, inplace=True)
    return ans


def test_best_models(data_X, data_y, comment):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=1337)

    # See how well the models do
    knn = KNeighborsClassifier(n_neighbors=5, leaf_size=10, metric='minkowski', p=1)
    knn.fit(X_train, y_train)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    print('Data "{}" ({}) was tested by the best models. Results are: {}, {}, {}'.format(
        comment, len(X_test.columns), knn.score(X_test, y_test), rf.score(X_test, y_test), dt.score(X_test, y_test)))


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
    #distinct_rows(data)

    data_y = data['class_e']
    data_X = data.drop(['class_e'], axis=1)


    comment = "Nominal"
    data_X_trim = data_X
    create_decision_tree(data_X_trim, data_y)
    test_best_models(data_X_trim, data_y, comment)

    comment = "Removed attributes deemed not useful by regressions"
    data_X_trim = trim_columns(data_X, ['odor', 'stalk-root', 'spore-print-color'])
    create_decision_tree(data_X_trim, data_y)
    test_best_models(data_X_trim, data_y, comment)

    comment = "Only 4 easy regression-decided attributes"
    data_X_trim = trim_columns(data_X, ['stalk-shape', 'cap-shape', 'gill-color', 'stalk-root'])
    create_decision_tree(data_X_trim, data_y)
    test_best_models(data_X_trim, data_y, comment)