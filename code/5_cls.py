import os
os.chdir(r"D:\My_Drive\MA_semester04\Tzu\paper\do")
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, linear_model
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Digraph
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
# Need to change path




data_ml = r'D:\My_Drive\MA_semester04\Tzu\paper\wdata\mu_ML.dta'
df = pd.read_stata(data_ml)



data_pd = r'D:\My_Drive\MA_semester04\Tzu\paper\wdata\mu_PD.dta'
df_pd = pd.read_stata(data_pd)
X_pd = df_pd[['gender',  'age',  "kid03", "kid35", "kid615", "kid1517", "kid18",
              "edu1", "edu2", "edu3", "edu4", "edu5", "edu6", "edu7", "edu8", "edu9",
              "rel1", "rel2", "rel3", "rel4", "rel5", "rel6", "rel7", "rel8", "rel9", "rel10", "rel11", "rel12", "rel13", "rel14",
              "mar1", "mar2", "mar3", "mar4",
              "county1", "county2", "county3", "county4", "county5", "county6", "county7", "county8", "county9",
              "county10", "county11", "county12", "county13", "county14", "county15", "county16", "county17",
              "county18", "county19", "county20",
              "major1", "major2", "major3", "major4", "major5", "major6", "major7", "major8", "major9", "major10", "major11"]]
# display(df.head())
# df = df.drop(df[df['treat']==2].index)
X = df[['gender', 'age', "kid03", "kid35", "kid615", "kid1517", "kid18",
        "edu1", "edu2", "edu3", "edu4", "edu5", "edu6", "edu7", "edu8", "edu9",
        "rel1", "rel2", "rel3", "rel4", "rel5", "rel6", "rel7", "rel8", "rel9", "rel10", "rel11", "rel12", "rel13", "rel14",
        "mar1", "mar2", "mar3", "mar4",
        "county1", "county2", "county3", "county4", "county5", "county6", "county7", "county8", "county9",
        "county10", "county11", "county12", "county13", "county14", "county15", "county16", "county17",
        "county18", "county19", "county20",
        "major1", "major2", "major3", "major4", "major5", "major6", "major7", "major8", "major9", "major10", "major11"]]
y = df['treat']
#X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

# split X into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# ------------------------------------------------------------------------------------------ Tree
# Tree
# criterion : impurity function
# max_depth : maximum depth of tree
# random_state : seed of random number generator
tree = DecisionTreeClassifier( criterion='entropy',
                               splitter='best',
                               max_depth=5,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               min_weight_fraction_leaf=0.0,
                               max_features='sqrt',
                               random_state=0,
                               max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               class_weight=None,
                               ccp_alpha=0.0)

tree.fit(X_train, y_train)

yt_pred_tree = tree.predict(X_train)
y_pred_tree = tree.predict(X_test)

print('Accuracy (tree,  train): %.2f' % accuracy_score(y_train, yt_pred_tree))
print('Accuracy (tree, test): %.2f' % accuracy_score(y_test, y_pred_tree))

plot_confusion_matrix(tree, X_test, y_test)
plt.savefig(r'D:\My_Drive\MA_semester04\Tzu\paper\pic\confusion_tree.png', dpi=300)
plt.show()

if not os.path.exists("./output/") : os.mkdir("./output/")
export_graphviz(
    tree,
    out_file='./output/tree.dot',
    feature_names=X.columns.values
)

# cmd
# dot -Tpng D:\REPL_pkg\TW\Py\output\tree.dot -o D:\REPL_pkg\TW\Py\output\fig-tree.png
#
bagging = BaggingClassifier(n_estimators=100,
                            random_state=0,
                            max_samples=65705,
                            max_features=20)
bagging.fit(X_train, y_train)
yt_pred_bagging = bagging.predict(X_train)
y_pred_bagging = bagging.predict(X_test)
print('Accuracy (bagging, train): %.2f' % accuracy_score(y_train, yt_pred_bagging))
print('Accuracy (bagging, test): %.2f' % accuracy_score(y_test, y_pred_bagging))
plot_confusion_matrix(bagging, X_test, y_test)
plt.savefig(r'D:\My_Drive\MA_semester04\Tzu\paper\pic\confusion_forest.png', dpi=300)
plt.show()
#------------------------------------------------------------------------------------------- Random Forests
forest = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features='sqrt',
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=None,
                                random_state=0,
                                verbose=0,
                                warm_start=False,
                                class_weight=None,
                                ccp_alpha=0.0,
                                max_samples=None)
forest.fit(X_train, y_train)

yt_pred_forest = forest.predict(X_train)
y_pred_forest = forest.predict(X_test)
print('Accuracy (forest, train): %.2f' % accuracy_score(y_train, yt_pred_forest))
print('Accuracy (forest, test): %.2f' % accuracy_score(y_test, y_pred_forest))
plot_confusion_matrix(forest, X_test, y_test)
plt.savefig(r'D:\My_Drive\MA_semester04\Tzu\paper\pic\confusion_forest.png', dpi=300)
plt.show()
# Feature Importance
importances = forest.feature_importances_
# get sort indices in descending order
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            X.columns.values[indices[f]],
                            importances[indices[f]]))

plt.figure()
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center',
        alpha=0.5)

plt.xticks(range(X_train.shape[1]),
           X.columns.values[indices], rotation=90)
plt.tight_layout()
plt.savefig('./output/fig-forest-feature-importances.png', dpi=300)
plt.show()

"""
#
logi = LogisticRegression(penalty='l2',
                         dual=False,
                         tol=0.0001,
                         C=1.0,
                         fit_intercept=True,
                         intercept_scaling=1,
                         class_weight=None,
                         random_state=0,
                         solver='lbfgs',
                         max_iter=100,
                         multi_class='auto',
                         verbose=0,
                         warm_start=False,
                         n_jobs=None,
                         l1_ratio=None)
model_logi = logi.fit(X_train, y_train)

yt_pred_logi = logi.predict(X_train)
y_pred_logi = logi.predict(X_test)
print('Accuracy (forest, train): %.2f' % accuracy_score(y_train, yt_pred_logi))
print('Accuracy (forest, test): %.2f' % accuracy_score(y_test, y_pred_logi))
"""
#------------------------------------------------------------------------------------------- AdaBoost
# AdaBoost Model
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=100,
                         learning_rate=1.0,
                         random_state=0
                         )
# Train Adaboost Classifer
model_abc = abc.fit(X_train, y_train)

#Predict the response for test dataset
yt_pred_Ada = model_abc.predict(X_train)
y_pred_Ada = model_abc.predict(X_test)


print('Accuracy (AdaBoost, train): %.2f' % metrics.accuracy_score(y_train, yt_pred_Ada))
print('Accuracy (AdaBoost, test): %.2f' % metrics.accuracy_score(y_test, y_pred_Ada))
plot_confusion_matrix(model_abc, X_test, y_test)
plt.savefig(r'D:\My_Drive\MA_semester04\Tzu\paper\pic\confusion_Ada.png', dpi=300)
plt.show()
#------------------------------------------------------------------------------------------- Gradient Boosting
GBC = GradientBoostingClassifier(learning_rate=0.1,
                                 n_estimators=100,
                                 subsample=1.0,
                                 criterion='friedman_mse',
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 max_depth=3,
                                 min_impurity_decrease=0.0,
                                 init=None,
                                 random_state=0,
                                 max_features=None,
                                 verbose=0,
                                 max_leaf_nodes=None,
                                 warm_start=False,
                                 validation_fraction=0.1,
                                 n_iter_no_change=None,
                                 tol=0.0001,
                                 ccp_alpha=0.0)
model_GBC = GBC.fit(X_train, y_train)
yt_pred_GBC = model_GBC.predict(X_train)
y_pred_GBC = model_GBC.predict(X_test)

print('Accuracy (GradientBoosting, train): %.2f' % metrics.accuracy_score(y_train, yt_pred_GBC))
print('Accuracy (GradientBoosting, test): %.2f' % metrics.accuracy_score(y_test, y_pred_GBC))
plot_confusion_matrix(model_GBC, X_test, y_test)
plt.savefig(r'D:\My_Drive\MA_semester04\Tzu\paper\pic\confusion_GB.png', dpi=300)
plt.show()
#------------------------------------------------------------------------------------------- MLP
MLP = MLPClassifier(hidden_layer_sizes=(100,),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    power_t=0.5,
                    max_iter=500,
                    shuffle=True,
                    random_state=0,
                    tol=0.0001,
                    verbose=False,
                    warm_start=False,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-08,
                    n_iter_no_change=10,
                    max_fun=15000
                    )
model_MLP= MLP.fit(X_train, y_train)
yt_pred_MLP = model_MLP.predict(X_train)
y_pred_MLP = model_MLP.predict(X_test)
print('Accuracy (MLP, train): %.2f' % metrics.accuracy_score(y_train, yt_pred_MLP))
print('Accuracy (MLP, test): %.2f' % metrics.accuracy_score(y_test, y_pred_MLP))
plot_confusion_matrix(model_MLP, X_test, y_test)
plt.savefig(r'D:\My_Drive\MA_semester04\Tzu\paper\pic\confusion_MLP.png', dpi=300)
plt.show()
#------------------------------------------------------------

Y_test = label_binarize(y_test, classes=[*range(3)])
wid=0.05

precision, recall, thresholds = precision_recall_curve(Y_test[:, 0], tree.predict_proba(X_test)[:, 0])
plt.scatter(recall, precision, label="Decision Trees", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_tree_t = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

precision, recall, thresholds = precision_recall_curve(Y_test[:, 0], bagging.predict_proba(X_test)[:, 0])
plt.scatter(recall, precision, label="Bagging", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_bagging_t = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

precision, recall, thresholds = precision_recall_curve(Y_test[:, 0], forest.predict_proba(X_test)[:, 0])
plt.scatter(recall, precision, label="Random Forest", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_forest_t = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

precision, recall, thresholds = precision_recall_curve(Y_test[:, 0], model_abc.predict_proba(X_test)[:, 0])
plt.scatter(recall, precision, label="AdaBoost", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_ada_t = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

precision, recall, thresholds = precision_recall_curve(Y_test[:, 0], model_GBC.predict_proba(X_test)[:, 0])
plt.scatter(recall, precision, label="Gradient Boosting", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_gbc_t = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

precision, recall, thresholds = precision_recall_curve(Y_test[:, 0], model_MLP.predict_proba(X_test)[:, 0])
plt.scatter(recall, precision, label="Multi-layer Perceptron", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_multi_t = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right", markerscale=10)
plt.savefig(r'D:\My_Drive\MA_semester04\Tzu\paper\pic\PR_curve_fir_t.png', dpi=300)
plt.show()

precision, recall, thresholds = precision_recall_curve(Y_test[:, 1], tree.predict_proba(X_test)[:, 1])
plt.scatter(recall, precision, label="Decision Trees", s=wid )
thresholds = thresholds.tolist()
thresholds.append(1)
pr_tree_c = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

precision, recall, thresholds = precision_recall_curve(Y_test[:, 1], bagging.predict_proba(X_test)[:, 1])
plt.scatter(recall, precision, label="Bagging", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_bagging_c = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

precision, recall, thresholds = precision_recall_curve(Y_test[:, 1], forest.predict_proba(X_test)[:, 1])
plt.scatter(recall, precision, label="Random Forest", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_forest_c = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

precision, recall, thresholds = precision_recall_curve(Y_test[:, 1], model_abc.predict_proba(X_test)[:, 1])
plt.scatter(recall, precision, label="AdaBoost", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_ada_c = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

precision, recall, thresholds = precision_recall_curve(Y_test[:, 1], model_GBC.predict_proba(X_test)[:, 1])
plt.scatter(recall, precision, label="Gradient Boosting", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_gbc_c = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

precision, recall, thresholds = precision_recall_curve(Y_test[:, 1], model_MLP.predict_proba(X_test)[:, 1])
plt.scatter(recall, precision, label="Multi-layer Perceptron", s=wid)
thresholds = thresholds.tolist()
thresholds.append(1)
pr_multi_c = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right", markerscale=10)
plt.savefig(r'D:\My_Drive\MA_semester04\Tzu\paper\pic\PR_curve_fir_c.png', dpi=300)
plt.show()

df_pd['y_tree_t'] = tree.predict_proba(X_pd)[:, 0]
df_pd['y_bagging_t'] = bagging.predict_proba(X_pd)[:, 0]
df_pd['y_forest_t'] = forest.predict_proba(X_pd)[:, 0]
df_pd['y_Ada_t'] = model_abc.predict_proba(X_pd)[:, 0]
df_pd['y_GBC_t'] = model_GBC.predict_proba(X_pd)[:, 0]
df_pd['y_MLP_t'] = model_MLP.predict_proba(X_pd)[:, 0]

df_pd['y_tree_c'] = tree.predict_proba(X_pd)[:, 1]
df_pd['y_bagging_c'] = bagging.predict_proba(X_pd)[:, 1]
df_pd['y_forest_c'] = forest.predict_proba(X_pd)[:, 1]
df_pd['y_Ada_c'] = model_abc.predict_proba(X_pd)[:, 1]
df_pd['y_GBC_c'] = model_GBC.predict_proba(X_pd)[:, 1]
df_pd['y_MLP_c'] = model_MLP.predict_proba(X_pd)[:, 1]

df_pd['y_tree_o'] = tree.predict_proba(X_pd)[:, 2]
df_pd['y_bagging_o'] = bagging.predict_proba(X_pd)[:, 2]
df_pd['y_forest_o'] = forest.predict_proba(X_pd)[:, 2]
df_pd['y_Ada_o'] = model_abc.predict_proba(X_pd)[:, 2]
df_pd['y_GBC_o'] = model_GBC.predict_proba(X_pd)[:, 2]
df_pd['y_MLP_o'] = model_MLP.predict_proba(X_pd)[:, 2]

df_pd['y_tree'] = tree.predict(X_pd)
df_pd['y_bagging'] = bagging.predict(X_pd)
df_pd['y_forest'] = forest.predict(X_pd)
df_pd['y_Ada'] = model_abc.predict(X_pd)
df_pd['y_GBC'] = model_GBC.predict(X_pd)
df_pd['y_MLP'] = model_MLP.predict(X_pd)

df_pd.to_stata(r'D:\My_Drive\MA_semester04\Tzu\paper\wdata\mu_MLPD.dta', version=118)


