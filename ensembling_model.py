# submission gets 0.78468 public score
# coding: utf-8

# import modules
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
                             ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

# 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。
# %matplotlib inline
get_ipython().run_line_magic('matplotlib', 'inline')

# 计数器
from collections import Counter

# set default seaborn config
sns.set(style='white', context='notebook', palette='deep')

# Load data
train = pd.read_csv("/Users/yangxinchen/Documents/Python/python_wheel/\
                    machine_learning/kaggle_titanic/train.csv")
test = pd.read_csv("/Users/yangxinchen/Documents/Python/python_wheel/\
                    machine_learning/kaggle_titanic/test.csv")
IDtest = test["PassengerId"]


#Outlier detection 检测异常值
def detect_outliers(df, n, features):
    outlier_indices = []

    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range 四分位差
        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR
        print(outlier_step)
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + \
                        outlier_step)].index
        
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


# drop outliers
Outliers_to_drop = detect_outliers(train, 2, ["SibSp","Parch","Fare"])

# print dropped ones
train.loc[Outliers_to_drop]
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# joining train and test set
train_len = len(train)
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


#check for null and missing values
dataset = dataset.fillna(np.nan)
dataset.isnull().sum()

# print train set info
train.info()
train.isnull().sum()

# print train head rows
train.head()

# print train set data types
train.dtypes

# print train set description
train.describe()


# Feature analysis
# Numerical values， plot correlation heatmap
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),
                    annot=True, fmt=".2f", cmap="coolwarm")


# Explore SibSp feature vs Survived
g = sns.catplot(x="SibSp", y="Survived", data=train, 
                kind="bar", height=6, palette="muted")
g.despine(left=True)
g = g.set_ylabels("survial probability")


# Explore Parch feature vs Survived 
g = sns.catplot(x="Parch", y="Survived", data=train, 
                kind="bar", height=6, palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")

## test segment ##
# g = sns.catplot(x="Age", y="Survived", data=train, kind="bar", height=6, palette="muted")
# g.despine(left=True)
# g = g.set_ylabels("survival probability")

# Explore Age distribution
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], 
                    color="Red", shade=True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], 
                    color="Blue", shade=True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived", "Survived"])

# check null value in "Fare", and fill them with median number
dataset["Fare"].isnull().sum()
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

# Explore Fare distribution
g = sns.distplot(dataset["Fare"], color="m", label="Skewness: %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(dataset["Fare"], color="m", label="Skewness: %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# Categorical values processing
# Sex
g = sns.barplot(x="Sex", y="Survived", data=train)
g = g.set_ylabel("Survival Probability")

train[["Sex", "Survived"]].groupby('Sex').mean()


# Explore Pclass vs Survived
g = sns.catplot(x="Pclass", y="Survived", data=train, kind="bar", 
                height=6, palette="muted")
g.despine(left=True)
g = g.set_ylabels("Survival probability")

# Explore Pclass vs Survived by Sex
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train, 
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Survival probability")

# Embarked
dataset["Embarked"].isnull().sum()
dataset["Embarked"] = dataset["Embarked"].fillna("S")

# Explore Embarked vs Survived
g = sns.catplot(x="Embarked", y="Survived", data=train, 
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

# Explore Pclass vs Embarked
g = sns.catplot("Pclass", col="Embarked", data=train, 
                height=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# Filling missing values
# Age 
# Explore Age vs Sex, Parch, Pclass and SibSp
g = sns.catplot(y="Age", x="Sex", data=dataset, kind="box")
g = sns.catplot(y="Age", x="Sex", hue="Pclass", data=dataset, kind="box")
g = sns.catplot(y="Age", x="Parch", data=dataset, kind="box")
g = sns.catplot(y="Age", x="SibSp", data=dataset, kind="box")

# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male":0, "female":1})

# plot correlation heatmap
g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(), cmap="BrBG", annot=True)


# Filling missing value of Age
# Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age: 
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & \
                                (dataset['Parch'] == dataset.iloc[i]["Parch"]) & \
                                (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i] = age_pred
    else:
        dataset['Age'].iloc[i] = age_med
            
# plot relations between Survived and Age 
g = sns.catplot(x="Survived", y="Age", data=train, kind="box")
g = sns.catplot(x="Survived", y="Age", data=train, kind="violin")


# Feature engineering 
# Name/Title, check data type for "Name" feature
dataset["Name"].head()

# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()

# plot numbers of each Title
g = sns.countplot(x="Title", data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)

# handling with special Title and turn it into numberic feature
dataset["Title"] = dataset["Title"].replace(['Lady','the Countess','Countess','Capt',
                    'Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms":1, "Mme":1, 
                    "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

# set data type as int 
dataset["Title"] = dataset["Title"].astype(int)

# print "Title" head rows
dataset["Title"].head()

# plot number counts for updated "Title"
g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])

# plot relations between updated "Title" and "Survived" col
g = sns.catplot(x="Title", y="Survived", data=dataset, kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# Drop Name variable
dataset.drop(labels=["Name"], axis=1, inplace=True)


# Family size (merge features)
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
# plot Survival Probability with Family size
g = sns.factorplot(x="Fsize", y="Survived", data=dataset)
g = g.set_ylabels("Survival Probability")

# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if s == 2 else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s>=5 else 0)

# plot Survival Probability with different family size
g = sns.catplot(x="Single", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="SmallF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="MedF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="LargeF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")


# convert to indicator values Title and Embarked
dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")

# print head rows for check
dataset.head()


# handling with Cabin
dataset["Cabin"].head()
dataset["Cabin"].describe()
dataset["Cabin"].isnull().sum()
dataset["Cabin"][dataset["Cabin"].notnull()].head()

# Replace the Cabin number by the type of cabin 'X' if not 
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])

# plot the number counts in "Cabin"
g = sns.countplot(dataset["Cabin"], order=['A','B','C','D','E','F','G','T','X'])

# plot the Survival Probability for different Cabin
g = sns.catplot(y="Survived", x="Cabin", data=dataset, 
                kind="bar", order=['A','B','C','D','E','F','G','T','X'])
g = g.set_xlabels("survival probability")

# set one-hot encode for "Cabin"
dataset = pd.get_dummies(dataset, columns=["Cabin"], prefix="Cabin")


# handling with Ticket
dataset["Ticket"].head()
# Treat Ticket by extracting the tikcet prefix. When there is no prefix it returns X.
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append('X')

dataset["Ticket"] = Ticket
dataset["Ticket"].head()

# set dummies for "Ticket" columns
dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")


# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")


# Drop PassengerId
dataset.drop(labels=["PassengerId"], axis=1, inplace=True)


# check dataset head
dataset.head()



## MODELING ##
## Seperate train dataset and test dataset
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"],axis=1, inplace=True)

# Separate train features and label
train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"], axis=1)


# Simple modeling
# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)

# Modeling step Test differents algorithms
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
                                            random_state=random_state, learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
#classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))
#classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y=Y_train, 
                            scoring="accuracy", cv=kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

# cv_res = pd.DataFrame({"CrossValMeans":cv_means, "CrossValerrors":cv_std, "Algorithm":
# ["SVC","DecistionTree","AdaBoost","RandomForest", "ExtraTrees","GradientBoosting", 
# "MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinerDiscriminantAnalysis"]})
cv_res = pd.DataFrame({"CrossValMeans":cv_means, "CrossValerrors":cv_std, "Algorithm":\
    ["SVC","DecistionTree","AdaBoost","RandomForest", "ExtraTrees","GradientBoosting", \
    "KNeighboors","LogisticRegression"]})


# plot mean Accuracy for each classifier
g = sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", 
                orient="h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")



# Hyperparameter tunning for best models
# META MODELING WITH ADABOOST, RF, EXTRATREES AND GRADIENTBOOSTING

# Adaboost
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion":["gini", "entropy"],
                 "base_estimator__splitter":["best", "random"],
                 "algorithm":["SAMME", "SAMME.R"],
                 "n_estimators": [1, 2],
                 "learning_rate": [0.0001, 0.001, 0.1, 0.2, 0.3, 1.5]}

gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, 
                            scoring="accuracy", n_jobs=4, verbose=1)
gsadaDTC.fit(X_train, Y_train)
ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_


#ExtraTrees
ExtC = ExtraTreesClassifier()
ex_param_grid = {"max_depth": [None],
                "max_features":[1, 3, 10],
                "min_samples_split":[2, 3, 10],
                "min_samples_leaf":[1, 3, 10],
                "bootstrap":[False],
                "n_estimators":[100, 300],
                "criterion":["gini"]}
gsExtC = GridSearchCV(ExtC, param_grid=ex_param_grid, cv=kfold, 
                            scoring="accuracy", n_jobs=4, verbose=1)
gsExtC.fit(X_train, Y_train)
ExtC_best = gsExtC.best_estimator_

gsExtC.best_score_


# RFC parameters tunning
RFC = RandomForestClassifier()
rf_param_grid = {"max_depth":[None],
                "max_features":[1, 3, 10],
                "min_samples_split":[2, 3, 10],
                "min_samples_leaf":[1, 3, 10],
                "bootstrap":[False],
                "n_estimators":[100, 300],
                "criterion":["gini"]}

gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, 
                        scoring="accuracy", n_jobs=4, verbose=1)
gsRFC.fit(X_train, Y_train)
RFC_best = gsRFC.best_estimator_

gsRFC.best_score_



# GradientBoosting
GBC = GradientBoostingClassifier()
gb_param_grid = {"loss": ["deviance"],
                "n_estimators":[100, 200, 300],
                "learning_rate":[0.1, 0.05, 0.01],
                "max_depth":[4, 8],
                "min_samples_leaf":[100, 150],
                'max_features':[0.3, 0.1]}

gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, 
                        scoring="accuracy", n_jobs=4, verbose=1)
gsGBC.fit(X_train, Y_train)
GBC_best = gsGBC.best_estimator_

gsGBC.best_score_


# SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel':['rbf'],
                 'gamma':[0.001, 0.01, 0.1, 1],
                 'C':[1, 10, 50, 100, 200, 300, 1000]}
gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, 
                            scoring="accuracy", n_jobs=4, verbose=1)
gsSVMC.fit(X_train, Y_train)
SVMC_best = gsSVMC.best_estimator_

gsSVMC.best_score_


# Plot learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                       n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, 
                                cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', 
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
            label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_, 
                        "RF learning curves", X_train, Y_train, cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_, 
                        "ExtraTrees learning curves", X_train, Y_train, cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_, 
                        "SVM learning curves", X_train, Y_train, cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_, 
                        "AdaBoost learning curves", X_train, Y_train, cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_, 
                        "GradientBoosting learning curves", X_train, Y_train, cv=kfold)


# Feature importance of tree based classifiers
nrows = ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex="all", figsize=(15,15))
names_classifiers =[("AdaBoosting",ada_best),("ExtraTrees",ExtC_best),\
                    ("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40], 
            x=classifier.feature_importances_[indices][:40], orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance", fontsize=12)
        g.set_ylabel("Features", fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + "feature importance")
        nclassifier += 1


test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVMC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")

ensemble_results = pd.concat([test_Survived_RFC, test_Survived_ExtC, 
    test_Survived_SVMC, test_Survived_AdaC, test_Survived_GBC], axis=1)

g = sns.heatmap(ensemble_results.corr(), annot=True)


# voting classifier
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best), 
    ('svc', SVMC_best), ('adac', ada_best), ('gbc', GBC_best)], voting='soft', n_jobs=4)
votingC = votingC.fit(X_train, Y_train)


# generate submission
test_Survived = pd.Series(votingC.predict(test), name="Survived")
results = pd.concat([IDtest, test_Survived], axis=1)
results.to_csv("/Users/yangxinchen/Documents/Python/python_wheel/machine_learning/\
    kaggle_titanic_top4per/ensemble_python_voting.csv", index=False)

