import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle


# Was useful for viewing some files
from sklearn.tree import DecisionTreeClassifier


# Was interesting to look at some rules
# But ultimately, a dead end for our purpose
def apriori_apply(data):
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    freq_itemsets = apriori(data, min_support=0.25, use_colnames=True)
    ar = association_rules(freq_itemsets, metric='confidence', min_threshold=0.5)
    ar.sort_values('lift', inplace=True, ascending=False)
    print(ar)


# A must-have check
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
    graph.render('./trees/'+ comment,view=True)  # Creates a file


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


def validate_best_models(data_X, data_y, comment):
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(data_X, data_y, random_state=1337)

    # See how well the models do
    knn = KNeighborsClassifier(n_neighbors=9, leaf_size=10, metric='minkowski', p=1)
    knn.fit(X_train, y_train)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    print('Data "{}" ({}) was validated by the best models. Results are: {}, {}, {}'.format(
        comment, len(X_val.columns), knn.score(X_val, y_val), rf.score(X_val, y_val), dt.score(X_val, y_val)))


# Calculate and show ROC
def show_roc(data_X, data_y):
    rf = RandomForestClassifier()
    rf.fit(data_X, data_y)

    probs = pd.DataFrame(data=rf.predict_proba(data_X))
    probs = probs[1]
    cutoffs = pd.DataFrame({'cutoff': probs.unique()})
    cutoffs = cutoffs.sort_values(by='cutoff', axis=0)
    tpr = cutoffs.apply(lambda cut: numpy.sum(numpy.logical_and(probs >= cut[0], data_y == 1)) / numpy.sum(data_y == 1), axis=1)
    fpr = cutoffs.apply(lambda cut: numpy.sum(numpy.logical_and(probs >= cut[0], data_y == 0)) / numpy.sum(data_y == 0), axis=1)
    stats = pd.DataFrame({'cutoff': cutoffs.cutoff, 'tpr': tpr, 'fpr': fpr})
    import matplotlib.pyplot as plt
    plt.plot(stats.fpr, stats.tpr, 'b')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC of RandomForestClassifier")
    plt.show()


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

    # 100% accuracy! But not practical at all: people will need to type all the attributes by hand!
    comment = "Nominal"
    data_X_trim = data_X
    create_decision_tree(data_X_trim, data_y)
    validate_best_models(data_X_trim, data_y, comment)

    # Only 3 useful attributes lead to about 99% accuracy! That's great... But thinking realistically,
    # novice shroomers will be confused, because odor and spore-print-color are somewhat hard to accurately describe
    comment = "Removed attributes deemed not useful by regressions"
    data_X_trim = trim_columns(data_X, ['odor', 'stalk-root', 'spore-print-color'])
    create_decision_tree(data_X_trim, data_y)
    validate_best_models(data_X_trim, data_y, comment)
    # Why not just try aall possible combinations? There are too many, so we use regressions to heavily cut the number
    # we actually need to check

    # That's what we are after! Only 95% accuracy, but those 3 attributes are much easier to explain
    # Initially there was also cap-shape attribute, but it turned out that it is not necessary
    # Trees are ugly! https://github.com/scikit-learn/scikit-learn/pull/12866
    comment = "Only 3 easy regression-decided attributes"
    data_X_trim = trim_columns(data_X, ['stalk-shape', 'gill-color', 'stalk-root'])
    create_decision_tree(data_X_trim, data_y)
    validate_best_models(data_X_trim, data_y, comment)

    # Lets check how often every value occurs:
    print(' ', end='')
    for c in data_X_trim.columns:
        print(c + " " + str(sum(data_X_trim[c])), end=', ')
    print('')

    # Improved variant, with less options to make user interface even simpler
    # This will be used in the application
    comment = "Full easy mode"
    data_X_trim.drop(['gill-color_o', 'gill-color_e', 'gill-color_y', 'gill-color_r'], axis=1, inplace=True)
    create_decision_tree(data_X_trim, data_y)
    validate_best_models(data_X_trim, data_y, comment)

    print('\nEasy mode columns: ')
    print(data_X_trim.columns)

    # Pickling good model to be used in our application
    rf = RandomForestClassifier()
    rf.fit(data_X_trim, data_y)
    file = open('./pygame/dt-model.pickle', 'wb')
    pickle.dump(rf, file)

    # Calculate ROC
    show_roc(data_X_trim, data_y)