
import warnings
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings('ignore')

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def load_titanic_fnc():
    df = pd.read_csv(r'E:\PROJECTS\EDA_DATAPREPROCESSING_FUTUREENG\datasets\train.csv')
    return df
# GENERAL

df = load_titanic_fnc()

print(df.head())

print(df.tail())

print(df.info())

print(df.columns)

print(df.shape)

print(df.index)

print(df.describe().T)

print(df.isnull().values.any())

print(df.isnull().sum().sort_values(ascending=False))

# 2. CATEGORICAL VARIABLE ANALYSIS
print(df.Survived.unique())
print(df.Survived.value_counts())

# WHAT ARE THE NAMES OF CATEGORICAL VARIABLES?
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Categorical Variable count: ', len(cat_cols))
print(cat_cols)

# HOW MANY CLASSES DO CATEGORICAL VARIABLES HAVE?
print(df[cat_cols].nunique())

# COUNTPLOT CATEGORICAL VARIABLES
for col in cat_cols:
    sns.countplot(x=col, data=df)
    plt.show()


def cats_summary(data):
    cats_names = [col for col in data.columns if len(data[col].unique()) < 10 and data[col].dtypes == 'O']
    for var in cats_names:
        print(pd.DataFrame({var: data[var].value_counts(),
                            "Ratio": 100 * data[var].value_counts() / len(data)}), end="\n\n\n")
        sns.countplot(x=var, data=data)
        plt.show()


cats_summary(df)


def cats_summary(data, categorical_cols, number_of_classes=10):
    var_count = 0  # count of categorical variables will be reported
    vars_more_classes = []  # categorical variables that have more than a number specified.
    for var in data:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # choose according to class count
                print(pd.DataFrame({var: data[var].value_counts(),
                                    "Ratio": 100 * data[var].value_counts() / len(data)}),
                      end="\n\n\n")
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


cats_summary(df, cat_cols)

# 3. NUMERICAL VARIABLE ANALYSIS

# GENERAL
print(df.describe().T)
print(df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T)

# NUMERICAL VARIABLES COUNT OF DATASET?
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and
            col not in "PassengerId" and
            col not in "Survived"]
print('Numerical Variables Count: ', len(num_cols))
print('Numerical Variables: ', num_cols)


# Histograms for numerical variables?
def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


hist_for_nums(df, num_cols)

# 4. TARGET ANALYSIS

# DISTRIBUTION OF "SURVIVED" VARIABLE
print(df["Survived"].value_counts())


def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target);
    facet.add_legend()


# TARGET ANALYSIS BASED ON CATEGORICAL VARIABLES
def target_summary_with_cat(data, target):
    cats_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in target]
    for var in cats_names:
        print(pd.DataFrame({"TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")
        plot_categories(df, cat=var, target='Survived')
        plt.show()


target_summary_with_cat(df, "Survived")


# TARGET ANALYSIS BASED ON NUMERICAL VARIABLES
def target_summary_with_nums(data, target):
    num_names = [col for col in data.columns if len(data[col].unique()) > 5
                 and df[col].dtypes != 'O'
                 and col not in target
                 and col not in "PassengerId"]

    for var in num_names:
        print(df.groupby(target).agg({var: np.mean}), end="\n\n\n")


target_summary_with_nums(df, "Survived")


# 5.SAYISAL DEGISKENLERIN BIRBIRLERINE GORE INCELENMESI

def correlation_matrix(df):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[num_cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show()

df.corr()

correlation_matrix(df)

# 6. WORK WITH OUTLIERS

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1 * interquantile_range
    low_limit = quartile1 - 0.3 * interquantile_range
    return low_limit, up_limit


num_cols2 = [col for col in df.columns if df[col].dtypes != 'O' and
             col not in "PassengerId" and
             col not in "Survived"
             and len(df[col].unique()) > 10]

# def Has_outlieers(data, column):
#     low, high = outlier_thresholds(df, col)
#     if (df[(df[col] < low) | (df[col] > high)].shape[0] > 0):
#         print(col, 'outlier status' ' :yes')


# for col in num_cols2:
#     Has_outlieers(df, num_cols2)


# 1. Report outliers
# 2. plot a boxplot of variables that have outliers
# 2. plot or do not plot will be optional
# 3. variables that have outliers will be returned as a list

def Has_outliers(data, number_col_names, plot=False):
    Outlier_variable_list = []

    for col in number_col_names:
        low, high = outlier_thresholds(df, col)

        if (df[(data[col] < low) | (data[col] > high)].shape[0] > 0):
            Outlier_variable_list.append(col)
            if (plot == True):
                sns.boxplot(x=col, data=df)
                plt.show()
    print('Variables that has outliers: ', Outlier_variable_list)
    return Outlier_variable_list


def Replace_with_thresholds(data, col):
    low, up = outlier_thresholds(data, col)
    data.loc[(data[col] < low), col] = low
    data.loc[(data[col] > up), col] = up
    print("Outliers for ", col, "column have been replaced with thresholds ",
          low, " and ", up)

#
var_names = Has_outliers(df, num_cols2)

# print(var_names)

for col in var_names:
    Replace_with_thresholds(df, col)

Has_outliers(df, num_cols2)


# 7. MISSING VALUE ANALYSIS

# Is there any missing values
print(df.isnull().values.any())

# Missing value counts of variables
print(df.isnull().sum().sort_values(ascending=False))

# missing value visualisation
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

#  value counts of variables
print(df.notnull().sum())

# Total count of missing variables in dataset
print(df.isnull().sum().sum())

# Samples have at least one missing value
print(df[df.isnull().any(axis=1)])

# Samples do not have any missing value
print(df[df.notnull().all(axis=1)])

# Missing value ratio
print((df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False))

cols_have_missing = [col for col in df.columns if df[col].isnull().sum() > 0]

print(cols_have_missing)
#

def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na

#
cols_with_na = missing_values_table(df)
#
# print(cols_with_na)
#
# df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
# df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

# def missing_values_table(dataframe):
#     variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
#     n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
#     ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
#     missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
#     print(missing_df)
#     return variables_with_na


missing_values_table(df)

def feature_median(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Survived']].groupby(['Survived'])[[var]].median().reset_index()
    return temp

#Age
feature_median('Age')

df.loc[(df['Survived'] == 0 ) & (df['Age'].isnull()), 'Age'] = 28.0
df.loc[(df['Survived'] == 1 ) & (df['Age'].isnull()), 'Age'] = 28.0

df["Age"] = df["Age"].astype('int')

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()


# RELATIONSHIP AMONG MISSING VALUES AND TARGET VARIABLE

def missing_vs_target(dataframe, target, variable_with_na):
    temp_df = dataframe.copy()

    for variable in variable_with_na:
        temp_df[variable + '_NA_FLAG'] = np.where(temp_df[variable].isnull(), 1, 0)

    flags_na = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for variable in flags_na:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(variable)[target].mean()}), end="\n\n\n")


print(cols_with_na)

missing_vs_target(df, "Survived", cols_with_na)

df.drop("Cabin", axis=1, inplace=True)

df.isnull().values.any()

# 11. FEATURE ENGINEERING

# IS_ALONE
df.loc[((df['SibSp'] + df['Parch']) > 0), "IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "IS_ALONE"] = "YES"

# Is DR?
df["IS_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

print("Doktor sayisi: ", len(df[df["IS_DR"] == 1]))
print(df.groupby("IS_DR").agg({"Survived": "mean"}))

# TITLE FEATURE
df['TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df.head()

# the size of families (including the passenger)
df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

# OTHER FEATURES BASED ON FAMILY SIZE
df['Family_Single'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
df['Family_Small'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
df['Family_Large'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

df.head()

print("AGE ve SURVIVED kırılımlarinda Title\n",
      df[["TITLE", "Survived", "Age"]].groupby(["TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]}))

# NEW FEATURES RELATED WITH AGE AND SEX
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df["NEW_SEX_CAT"].value_counts()

df.columns
# 8. LABEL ENCODING

def label_encoder(dataframe):
    labelencoder = preprocessing.LabelEncoder()

    label_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                  and len(dataframe[col].value_counts()) == 2]

    for col in label_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe


df = label_encoder(df)


# 9. ONE-HOT ENCODING
def one_hot_encoder(dataframe, category_freq=20, nan_as_category=False):
    categorical_cols = [col for col in dataframe.columns if len(dataframe[col].value_counts()) < category_freq
                        and dataframe[col].dtypes == 'O']

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)

    return dataframe


df = one_hot_encoder(df)

print(df.head())


# 10. RARE ENCODING

def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in df.columns if df[col].dtypes == 'O'
                    and (df[col].value_counts() / len(df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(var)[target].mean()}), end="\n\n\n")


rare_analyser(df, "Survived", 0.01)


# Almost every value in Name and Ticket columns are rare values

def rare_encoder(dataframe, rare_perc):
    tempr_df = dataframe.copy()

    rare_columns = [col for col in tempr_df.columns if tempr_df[col].dtypes == 'O'
                    and (tempr_df[col].value_counts() / len(tempr_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = tempr_df[var].value_counts() / len(tempr_df)
        rare_labels = tmp[tmp < rare_perc].index
        tempr_df[var] = np.where(tempr_df[var].isin(rare_labels), 'Rare', tempr_df[var])

    return tempr_df


new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "Survived", 0.01)

df.columns

df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

# 10. STANDARDIZATION ???

print(df.head())

# drops after observing importances of features
df.drop(['IS_DR', 'Family_Single', 'Family_Large', 'TITLE_Col',
       'TITLE_Countess', 'TITLE_Don', 'TITLE_Dr', 'TITLE_Jonkheer',
       'TITLE_Lady', 'TITLE_Major', 'TITLE_Mlle',
       'TITLE_Mme', 'TITLE_Ms', 'TITLE_Rev',
       'TITLE_Sir', 'NEW_SEX_CAT_maturemale', 'NEW_SEX_CAT_seniorfemale',
       'NEW_SEX_CAT_seniormale'], axis=1, inplace=True)

# Modeling

y = df["Survived"]
X = df.drop(["Survived"], axis=1)

models = [#('RF', RandomForestClassifier())]
     ('XGB', GradientBoostingClassifier())]
        #  ("LightGBM", LGBMClassifier())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print('Base: ', msg)

    # RF Tuned
    if name == 'RF':
        rf_params = {"n_estimators": [200, 500, 1000, 1500],
                     "max_features": [5, 10, 50, 100],
                     "min_samples_split": [5, 10, 20, 50, 100],
                     "max_depth": [5, 10, 20, 50, None]}

        rf_model = RandomForestClassifier(random_state=123)
        print('RF Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(rf_model,
                             rf_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, verbose=2, n_jobs=-1).fit(X, y)  # ???
        print('RF Bitis zamani: ', datetime.now())
        rf_tuned = RandomForestClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(rf_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('RF Tuned: ', msg)
        print('RF Best params: ', gs_cv.best_params_)

        # Feature Importance
        feature_imp = pd.Series(rf_tuned.feature_importances_,
                                index=X.columns).sort_values(ascending=False)

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title("Değişken Önem Düzeyleri")
        plt.show()
        plt.savefig('rf_importances.png')

    # LGBM Tuned
    elif name == 'LightGBM':
        lgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
        "n_estimators": [500, 1000, 1500],
        "max_depth": [3, 5, 8, 10, 20, 50],
        'num_leaves': [31, 50, 100]}

        lgbm_model = LGBMClassifier(random_state=123)
        print('LGBM Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(lgbm_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        print('LGBM Bitis zamani: ', datetime.now())
        lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(lgbm_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('LGBM Tuned: ', msg)
        print('LGBM Best params: ', gs_cv.best_params_)

        # Feature Importance
        feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                                index=X.columns).sort_values(ascending=False)

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title("Değişken Önem Düzeyleri")
        plt.show()
        plt.savefig('lgbm_importances.png')

    # XGB Tuned
    elif name == 'XGB':
        xgb_params = {#"colsample_bytree": [0.05, 0.1, 0.5, 1],
                      'max_depth': np.arange(1, 11),
                      'subsample': [0.5, 1, 5],
                      'learning_rate': [0.005, 0.01],
                      'n_estimators': [100, 500, 1000],
                      'loss': ['deviance', 'exponential']}

        xgb_model = GradientBoostingClassifier(random_state=123)

        print('XGB Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(xgb_model,
                             xgb_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        print('XGB Bitis zamani: ', datetime.now())
        xgb_tuned = GradientBoostingClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(xgb_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('XGB Tuned: ', msg)
        print('XGB Best params: ', gs_cv.best_params_)




# XGB
# Base:  XGB: 0.838402 (0.038245)
# XGB Baslangic zamani:  2020-11-06 22:00:57.335724
# Fitting 10 folds for each of 360 candidates, totalling 3600 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# XGB Bitis zamani:  2020-11-06 22:22:52.178318
# XGB Tuned:  XGB: 0.845156 (0.000000)
# XGB Best params:  {'learning_rate': 0.005, 'loss': 'exponential', 'max_depth': 5, 'n_estimators': 1000, 'subsample': 0.5}
#
# RF
# Base:  RF: 0.804782 (0.041375)
# RF Baslangic zamani:  2020-11-06 21:10:53.471340
# Fitting 10 folds for each of 400 candidates, totalling 4000 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# RF Bitis zamani:  2020-11-06 21:51:07.949829
# RF Tuned:  RF: 0.842921 (0.000000)
# RF Best params:  {'max_depth': 10, 'max_features': 10, 'min_samples_split': 20, 'n_estimators': 1500}
#
# LGBM
# Base:  LightGBM: 0.823820 (0.044834)
# LGBM Baslangic zamani:  2020-11-06 21:54:22.045719
# Fitting 10 folds for each of 162 candidates, totalling 1620 fits
# LGBM Bitis zamani:  2020-11-06 21:58:28.465062
# LGBM Tuned:  LightGBM: 0.844045 (0.000000)
# LGBM Best params:  {'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 500, 'num_leaves': 50}