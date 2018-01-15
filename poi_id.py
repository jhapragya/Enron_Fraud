#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from numpy import mean
from sklearn.cross_validation import train_test_split

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features

financial_features = ['salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees'] 

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi']

poi_label = ['poi']

features_list = poi_label + email_features + financial_features

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Total number of data points

print("Total number of data points: %i" %len(data_dict))

# Allocation across classes (POI/non-POI)

poi = 0
for user in data_dict:
    if data_dict[user]['poi'] == True:
       poi += 1
print("Total number of poi: %i" % poi)
print("Total number of non-poi: %i" % (len(data_dict) - poi))
       
# Number of features used



all_features = data_dict[data_dict.keys()[0]]


print("There are %i features for each person in the dataset, and %i features \
are used" %(len(all_features), len(features_list)))

print(all_features)




# Are there features with many missing values? etc.
missing_values = {}

for feature in all_features:
    missing_values[feature] = 0

for user in data_dict:
    for feature in data_dict[user]:
        if data_dict[user][feature] == "NaN":
            missing_values[feature] += 1

print("The number of missing values for each feature: ")
for feature in missing_values:
    print("%s: %i" %(feature, missing_values[feature]))

    
### Task 2: Remove outliers

import matplotlib.pyplot as plt

def scatterplot(data_set, feature_x, feature_y):
    """
    This function takes a dict, 2 strings, and shows a 2d plot of 2 features
    """
    data = featureFormat(data_set, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        plt.scatter( x, y )
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

# Visualize data to identify outliers
print(scatterplot(data_dict,'total_payments','total_stock_value'))
print(scatterplot(data_dict,'salary','bonus'))
print(scatterplot(data_dict,'from_poi_to_this_person','from_this_person_to_poi'))
print(scatterplot(data_dict,'total_payments','other'))



identity = []
for user in data_dict:
    if data_dict[user]['total_payments'] != "NaN":
        identity.append((user, data_dict[user]['total_payments']))
print("Outlier:")
print("pragya")
print(identity)
print(sorted(identity, key = lambda x: x[1], reverse=True)[0:4])

# Find persons whose financial features are all "NaN"
finan_dict = {}
for user in data_dict:
    finan_dict[user] = 0
    for feature in financial_features:
        if data_dict[user][feature] == "NaN":
            finan_dict[user] += 1
print(sorted(finan_dict.items(), key=lambda x: x[1], reverse = True)[0:4])

# Find persons whose email features are all "NaN"
email_nan_dict = {}
for person in data_dict:
    email_nan_dict[person] = 0
    for feature in email_features:
        if data_dict[person][feature] == "NaN":
            email_nan_dict[person] += 1
print(sorted(email_nan_dict.items(), key=lambda x: x[1],reverse = True)[0:1])

# Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("BAXTER JOHN C", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for user in my_dataset:
    msg_from_poi = my_dataset[user]['from_poi_to_this_person']
    to_msg = my_dataset[user]['to_messages']
    if msg_from_poi != "NaN" and to_msg != "NaN":
        my_dataset[user]['msg_from_poi_ratio'] = msg_from_poi/float(to_msg)
    else:
        my_dataset[user]['msg_from_poi_ratio'] = 0


    msg_to_poi = my_dataset[user]['from_this_person_to_poi']
    from_msg = my_dataset[user]['from_messages']
    if msg_to_poi != "NaN" and from_msg != "NaN":
        my_dataset[user]['msg_to_poi_ratio'] = msg_to_poi/float(from_msg)
    else:
        my_dataset[user]['msg_to_poi_ratio'] = 0


new_features_list = features_list + ['msg_to_poi_ratio', 'msg_from_poi_ratio']







## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Removing features with low variance: 
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(features)

#Removes all but the k highest scoring features
from sklearn.feature_selection import f_classif
k=7
selector = SelectKBest(f_classif, k=7)
selector.fit_transform(features, labels)
print("Best features:")
scores = zip(new_features_list[1:],selector.scores_)

#This returns the list of key-value pairs in the dictionary, sorted by value from highest to lowest:
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print sorted_scores

optimized_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:k]
print(optimized_features_list)



# Extract from dataset without new features

data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# Extract from dataset with new features
data = featureFormat(my_dataset, optimized_features_list + \
                     ['msg_to_poi_ratio', 'msg_from_poi_ratio'], \
                     sort_keys = True)
new_labels, new_features = targetFeatureSplit(data)
new_features = scaler.fit_transform(new_features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

import numpy as np
def tune_parameters(grid_search, features, labels, params, iters=100):
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=i)

         


        grid_search.fit(features_train, labels_train)
        predict = grid_search.predict(features_test)

        acc = acc + [accuracy_score(labels_test, predict)] 
        pre = pre + [precision_score(labels_test, predict)]
        recall = recall + [recall_score(labels_test, predict)]
    print "accuracy: {}".format(np.mean(acc))
    print "precision: {}".format(np.mean(pre))
    print "recall: {}".format(np.mean(recall))

    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))

from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()   
nb_param = {}
nb_grid_search = GridSearchCV(nb_clf, nb_param)

print("Evaluate naive bayes model")
tune_parameters(nb_grid_search, features, labels, nb_param)
#accuracy: 0.81511627907
#precision: 0.37000793344
#ecall: 0.363567460317   

print("Evaluate naive bayes model using dataset with new features")
tune_parameters(nb_grid_search, new_features, new_labels, nb_param)
#accuracy: 0.812093023256
#precision: 0.370769801335
#recall: 0.38301984127

from sklearn import svm
svm_clf = svm.SVC()
svm_param = {'kernel': ['rbf', 'linear', 'poly'], 'C': [0.1, 1, 10, 100, 1000],\
          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}    
svm_grid_search = GridSearchCV(svm_clf, svm_param)
print("Evaluate svm model")
tune_parameters(svm_grid_search, features, labels, svm_param)
#accuracy: 0.866976744186
#precision: 0.0858333333333
#recall: 0.035623015873
#kernel = 'poly',
#C = 1,
#gamma = 1,

from sklearn import tree
dt_clf = tree.DecisionTreeClassifier()
dt_param = {'criterion':('gini', 'entropy'),
'splitter':('best','random')}
dt_grid_search = GridSearchCV(dt_clf, dt_param)

print("Evaluate Decision Tree")
tune_parameters(dt_grid_search, features, labels, dt_param)
#accuracy: 0.810697674419
#precision: 0.292176767677
#recall: 0.291884920635
#splitter = 'random',
#criterion = 'gini', 
tune_parameters(dt_grid_search, new_features, new_labels, dt_param)
#accuracy_score: 0.821627906977
#precision: 0.32162987013
#recall: 0.347027777778
#splitter = 'best',
#criterion = 'gini',


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=10)
rf_param = {}
rf_grid_search = GridSearchCV(rf_clf,rf_param)

print("Evaluate Random Forest model")
tune_parameters(rf_grid_search, features, labels, rf_param)
#accuracy: 0.868372093023
#precision: 0.404666666667
#recall: 0.172182539683
tune_parameters(rf_grid_search, new_features, new_labels, rf_param)
#accuracy: 0.872325581395
#precision: 0.449166666667
#recall: 0.199698412698



from sklearn import linear_model
from sklearn.pipeline import Pipeline
lo_clf = Pipeline(steps=[
        ('scaler', preprocessing.StandardScaler()),
        ('classifier', linear_model.LogisticRegression())])
         
lo_param = {'classifier__tol': [1, 0.1, 0.01, 0.001, 0.0001], \
            'classifier__C': [0.1, 0.01, 0.001, 0.0001]}
lo_grid_search = GridSearchCV(lo_clf, lo_param)
print("Evaluate logistic regression model")
tune_parameters(lo_grid_search, features, labels, lo_param)
#accuracy: 0.861395348837
#precision: 0.387476190476
#recall: 0.189150793651
#classifier__tol = 1,
#classifier__C = 0.1,
print("Evaluate logistic regression model using dataset with new features")
tune_parameters(lo_grid_search, new_features, new_labels, lo_param)
#accuracy: 0.858139534884
#precision: 0.381452380952
#recall: 0.195833333333
#classifier__tol = 1,
#classifier__C = 0.1,





### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn import naive_bayes
clf = naive_bayes.GaussianNB()
final_features_list = optimized_features_list #+ ['msg_to_poi_ratio', 'msg_from_poi_ratio']

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, final_features_list)
