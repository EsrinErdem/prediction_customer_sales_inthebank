"""Today, you started working in the Bank. Your manager take you to a meeting about a new commercial product of Bank. In this meeting, Mr. Yıldız, the Marketing Manager of the Commercial segment, stated that “Bank is one of the leading banks and we developed a new product for commercial clients. We have been working for months. As a result of the studies, we believe that clients with over 5 million TL annual net sales are the most suitable to offer this new product.”

He also added:  “As you know, banks do not have current financial statement of each company. Without financial statements, we cannot get proper net sales information of customers. We can already reach suitable clients for the new product for those with current financials available in our databases. However, we want to be sure that we reach all targeted customers. So, we have to find a way to decide which one of these clients without financials have actually over 5 million TL net sales.”

Then, the Commercial Loan Allocation Manager stated that total risk and total limit amounts in the banking sector can be reached for Bank clients and this information could be used for estimation.

Mr. Yıldız asked, “If the clients has more than 5 million TL total loan balance in banking sector, may  their net sales be higher than 5 million TL too?”

Commercial Loan Allocation Manager Mrs. İlhan replied “Even though there are exceptions, generally it is correct. In addition, companies with more experience and having institutional culture are generally in the leading position in the sector. Therefore, they should be evaluated."

Your manager took the floor and introduced you to other managers. “We can develop the most suitable model by looking at the data” he said, adding that you can develop a model that would solve this problem. In the end, by adding “with data we can create the most appropriate solution.”

After meeting, you take related information from databases. The following information could be reached through the queries in the databases. Information are shown as empty (NULL) if no data is available.

* YEAR	Date of data
* Customer_num	Customer identification number
* Establishment_Date	Company establishment date
* Number_of_Emp	Number of employees
* Profit	Annual profit
* Sector	Sector that company operates
* Region	Geographic region
* Total Risk 	Total loan balance amount in the banking sector
* Total Limit	Total limit in the banking sector
* Sales	0 if Sales =< 5 million TL
* 1 if Sales > 5 million TL
* 3 if Sales is not available.
"""

from helpers.data_prep import *
from helpers.eda import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler,  RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit,cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import cross_validate
import xgboost as xgb
import lightgbm as lgb
###########################################
# Uyarıları kapatmak için
###########################################
import warnings
pd.set_option('display.max_rows', None)
warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
# pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###########################################
# DATANIN OKUNMASI
###########################################
dfilk = pd.read_csv("datasets/Data_Set_Final_Project.csv")
dfilk.head()

#########################################
# EDA
#########################################
dfilk.shape

# verideki sales i 3 olanlari datadan cikariyoruz.
dfilk_3 = dfilk[dfilk['Sales'] == 3]
dfilk = dfilk[~(dfilk['Sales'] == 3)]

# tekrar shape'e bakalim
dfilk.shape

# datanin kolonlarina bakalim
dfilk.columns

#tarih değişkeni objectten tarih formatına çevirilir.
dfilk['Establishment_Date'] = pd.to_datetime(dfilk['Establishment_Date'])

#datamizi genel olarak inceleyelim.
check_df(dfilk)

# kategorik ve numerik kolonlari belirleyelim.
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dfilk)

# bagimli degiskene gore oranlarina bakalim
rare_analyser(dfilk,'Sales',cat_cols)

# outlier analizi yapalim.
for col in num_cols:
    print(col, check_outlier(dfilk, col))

# numerik kolonlarin outlierlarina grafikle bakalim.
sns.boxplot(dfilk['Total_Limit'])
plt.show()

sns.boxplot(dfilk['Total_Risk'])
plt.show()

sns.boxplot(dfilk['Profit'])
plt.show()

sns.boxplot(dfilk['Number_of_Emp'])
plt.show()

# Describe ina bakalim
dfilk.describe().T

# kategorik degisken ozetlerine bakalim.
cat_summary(dfilk, "Sector", plot=True)

cat_summary(dfilk, "Sales", plot=True)

cat_summary(dfilk, "Region", plot=True)

# numerik degisken ozetlerine bakalim
for col in num_cols:
    num_summary(dfilk, col, plot=True)

#Kuruluş yılı eski olup satışı 5 milyon tl nin üzerinde olan firmalar ile benzerlik kurabiliyor muyuz buna bakacağız.
#satışları 1 olanları(5 milyon tl üzeri olanlar) getirdik.
dfilk_1 = dfilk[dfilk['Sales']==1]
dfilk_1=dfilk_1.sort_values('Number_of_Emp',ascending=False)
#satışları 1 olanların kuruluş yıllarına dolayısıyla köklü kuruluş olup olmadığına baktık.
dfilk_1.groupby(['Establishment_Date','Number_of_Emp'])['Sales'].count().head(20)


#Kuruluş yılı eski olup satışı 5 milyon tl nin üzerinde olan firmalar ile benzerlik kurabiliyor muyuz buna bakacağız.
#satışları 1 olanları(5 milyon tl üzeri olanlar) getirdik.
dfilk_0 = dfilk[dfilk['Sales']==0]
dfilk_1=dfilk_1.sort_values('Number_of_Emp',ascending=False)
#satışları 1 olanların kuruluş yıllarına dolayısıyla köklü kuruluş olup olmadığına baktık.
dfilk_0.groupby(['Establishment_Date','Number_of_Emp'])['Sales'].count().head(20)

# region isimlerine bakalim
dfilk['Region'].unique()

# numerik kolonlarin sales a gore durumlarina bakalim.
for col in num_cols:
    target_summary_with_num(dfilk, "Sales", col)

# korelasyonlara bakalim.
dfilk.corr()
f, ax = plt.subplots(figsize=[10, 10])
sns.heatmap(dfilk.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

###########################################
# DATANIN OKUNMASI
###########################################

# df = pd.read_csv("datasets/Final_Project.csv")
# df.head()

###########################################
# DATA_PREP
###########################################

def data_prep(dataframe):
    # tarih değişkeni object. Onu tarih formatına çevirilir.
    dataframe['Establishment_Date'] = pd.to_datetime(dataframe['Establishment_Date'])
    check_df(dataframe)
    df_3 = dataframe[dataframe['Sales'] == 3]

    df = dataframe[~(dataframe['Sales'] == 3)]

    check_df(df)
    return df, df_3

# df, df_3 = data_prep(df)

# df.head()
# df_3.head()


###########################################
# # **MISSING VALUE LARIN DOLDURULMASI**
###########################################

def missing_value(dataframe):
    dataframe["Total_Risk"] = dataframe["Total_Risk"].fillna(dataframe.groupby("Sales")["Total_Risk"].transform("median"))
    dataframe["Total_Limit"] = dataframe["Total_Limit"].fillna(dataframe.groupby("Sales")["Total_Limit"].transform("median"))
    dataframe["Number_of_Emp"] = dataframe["Number_of_Emp"].fillna(dataframe.groupby("Sales")["Number_of_Emp"].transform("median"))
    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
    check_df(dataframe)
    return dataframe

# df = missing_value(df)

#check_df(df)

###########################################
# FEATURE ENGINEERING
###########################################

def feature_engineering(dataframe):

    #TOTAL_RISK
    a = dataframe.groupby('Customer_num')['Total_Risk'].sum()
    a = a.reset_index()
    a.rename(columns={'Total_Risk': 'sum_of_risk'}, inplace=True)
    dataframe = dataframe.merge(a, 'outer')

    # TOTAL_LIMIT
    b = dataframe.groupby('Customer_num')['Total_Limit'].sum()
    b = b.reset_index()
    b.rename(columns={'Total_Limit': 'sum_of_limit'}, inplace=True)
    dataframe = dataframe.merge(b, 'outer')

    # TOTAL_PROFIT
    c = dataframe.groupby('Customer_num')['Profit'].sum()
    c = c.reset_index()
    c.rename(columns={'Profit': 'sum_of_profit'}, inplace=True)
    dataframe = dataframe.merge(c, 'outer')

    #YEAR
    current_year = dataframe['YEAR'].max() + 1

    dataframe['year_tenure'] = current_year - dataframe['YEAR']
    current_date = pd.to_datetime('2017-12-31 0:0:0')  # analiz tarihi belirlenir
    dataframe["days"] = (current_date - dataframe['Establishment_Date']).dt.days
    dataframe['Establishment_Tenure'] = dataframe["days"] / 365
    dataframe['TEtablisement+Nemployeur'] = dataframe['Establishment_Tenure'] + dataframe['Number_of_Emp']
    dataframe['TEtablisement*Nemployeur'] = dataframe['Establishment_Tenure'] * dataframe['Number_of_Emp']
    dataframe['total_risk_rate'] = dataframe['Total_Risk'] / dataframe['Total_Limit']
    dataframe['tyear+testablisment+nempl'] = dataframe['cus_tenure'] + dataframe['Establishment_Tenure'] + dataframe['Number_of_Emp']
    dataframe['tyear*testablisment*nempl'] = dataframe['cus_tenure'] * dataframe['Establishment_Tenure'] * dataframe['Number_of_Emp']
    dataframe['New_Establishment_Tenure'] = pd.cut(dataframe['Establishment_Tenure'], bins=[0, 10, 14, 20, 40, 60, 119],
                                            labels=['0-10', '11-14', '15-20', '21-40', '41-60', '61-119'])
    dataframe = dataframe.drop(['YEAR', 'Establishment_Date', "days"], axis=1)

    #NUM_OF_EMPLOYEE
    dataframe['Cat_Number_of_Emp'] = pd.cut(dataframe['Number_of_Emp'], bins=[0, 4, 7, 15, 200, 3333],
                                     labels=['0_4', '5_7', '8-15', '16-200', '201-3333'])

    # PROFIT
    bins = [dataframe['Profit'].min() - 1, 46000, 460000, 1100000, dataframe['Profit'].max()]
    labels = ['VERY_SMALL', 'SMALL', 'MIDDLE', 'HIGH']
    dataframe['NEW_PROFIT'] = pd.cut(dataframe['Profit'], bins, labels=labels).astype(str)

    # AGE_CATEGORY
    bins = [dataframe['cus_tenure'].min() - 1, 4, 6, dataframe['cus_tenure'].max()]
    labels = ['YOUNG', 'MIDDLE', 'MATURE']
    dataframe['NEW_YEAR_TENURE'] = pd.cut(dataframe['cus_tenure'], bins, labels=labels).astype(str)

    # YEAR_PROFIT
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'YOUNG') & ((dataframe['NEW_PROFIT'] == 'VERY_SMALL') | (
            dataframe['NEW_PROFIT'] == 'SMALL')), 'NEW_YEAR_PROFIT'] = 'YOUNG_SMALL_PROFIT'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'YOUNG') & (
                dataframe['NEW_PROFIT'] == 'MIDDLE'), 'NEW_YEAR_PROFIT'] = 'YOUNG_LOW_PROFIT'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'YOUNG') & (
                dataframe['NEW_PROFIT'] == 'HIGH'), 'NEW_YEAR_PROFIT'] = 'YOUNG_LOW_PROFIT'

    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'MIDDLE') & ((dataframe['NEW_PROFIT'] == 'VERY_SMALL') | (
            dataframe['NEW_PROFIT'] == 'SMALL')), 'NEW_YEAR_PROFIT'] = 'MIDDLE_SMALL_PROFIT'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'MIDDLE') & (
            dataframe['NEW_PROFIT'] == 'MIDDLE'), 'NEW_YEAR_PROFIT'] = 'MIDDLE_NORMAL_PROFIT'
    dataframe.loc[
        (dataframe['NEW_YEAR_TENURE'] == 'MIDDLE') & (
                    dataframe['NEW_PROFIT'] == 'HIGH'), 'NEW_YEAR_PROFIT'] = 'MIDDLE_NORMAL_PROFIT'

    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'MATURE') & ((dataframe['NEW_PROFIT'] == 'VERY_SMALL') | (
            dataframe['NEW_PROFIT'] == 'SMALL')), 'NEW_YEAR_PROFIT'] = 'MATURE_SMALL_PROFIT'
    dataframe.loc[
        (dataframe['NEW_YEAR_TENURE'] == 'MATURE') & (
                    dataframe['NEW_PROFIT'] == 'MIDDLE'), 'NEW_YEAR_PROFIT'] = 'MATURE_HIGH_PROFIT'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'MATURE') & (
                dataframe['NEW_PROFIT'] == 'HIGH'), 'NEW_YEAR_PROFIT'] = 'MATURE_HIGH_PROFIT'

    # YEAR_SECTOR
    dataframe.loc[
        (dataframe['NEW_YEAR_TENURE'] == 'YOUNG') & (
                dataframe['Sector'] == 'MANUFACTURING'), 'NEW_YEAR_SECTOR'] = 'YOUNG_MANUFACTURING'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'YOUNG') & (
            dataframe['Sector'] == 'RETAIL-WHOLESALE'), 'NEW_YEAR_SECTOR'] = 'YOUNG_RETAIL-WHOLESALE'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'YOUNG') & (
                dataframe['Sector'] == 'SERVICES'), 'NEW_YEAR_SECTOR'] = 'YOUNG_SERVICES'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'YOUNG') & (
                dataframe['Sector'] == 'OTHERS'), 'NEW_YEAR_SECTOR'] = 'YOUNG_OTHERS'

    dataframe.loc[
        (dataframe['NEW_YEAR_TENURE'] == 'MIDDLE') & (
                dataframe['Sector'] == 'MANUFACTURING'), 'NEW_YEAR_SECTOR'] = 'MIDDLE_MANUFACTURING'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'MIDDLE') & (
            dataframe['Sector'] == 'RETAIL-WHOLESALE'), 'NEW_YEAR_SECTOR'] = 'MIDDLE_RETAIL-WHOLESALE'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'MIDDLE') & (
                dataframe['Sector'] == 'SERVICES'), 'NEW_YEAR_SECTOR'] = 'MIDDLE_SERVICES'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'MIDDLE') & (
                dataframe['Sector'] == 'OTHERS'), 'NEW_YEAR_SECTOR'] = 'MIDDLE_OTHERS'

    dataframe.loc[
        (dataframe['NEW_YEAR_TENURE'] == 'MATURE') & (
                dataframe['Sector'] == 'MANUFACTURING'), 'NEW_YEAR_SECTOR'] = 'MATURE_MANUFACTURING'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'MATURE') & (
            dataframe['Sector'] == 'RETAIL-WHOLESALE'), 'NEW_YEAR_SECTOR'] = 'MATURE_RETAIL-WHOLESALE'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'MATURE') & (
                dataframe['Sector'] == 'SERVICES'), 'NEW_YEAR_SECTOR'] = 'MATURE_SERVICES'
    dataframe.loc[(dataframe['NEW_YEAR_TENURE'] == 'MATURE') & (
                dataframe['Sector'] == 'OTHERS'), 'NEW_YEAR_SECTOR'] = 'MATURE_OTHERS'

    dataframe.columns = [col.upper() for col in dataframe.columns]
    return dataframe

# df = feature_engineering(df)
#
# df.head()
# df.columns
# df.shape
###########################################
#  One-Hot Encoding
###########################################

def one_hot(dataframe):
    ohe_cols = [col for col in dataframe.columns if 13 >= dataframe[col].nunique() >= 2 and col != 'SALES']
    dataframe = one_hot_encoder(dataframe, ohe_cols, drop_first=True)
    return dataframe

# ohe_df = one_hot(df)


# ohe_df.head()
# check_df(ohe_df)

#cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(ohe_df)

###########################################
# RobustScaler: Medyanı çıkar iqr'a böl.
###########################################
def robust_scaler(dataframe):
    rs = RobustScaler()

    dataframe["SUM_OF_RISK"] = rs.fit_transform(dataframe[['SUM_OF_RISK']])
    dataframe["SUM_OF_LIMIT"] = rs.fit_transform(dataframe[['SUM_OF_LIMIT']])
    dataframe["SUM_OF_PROFIT"] = rs.fit_transform(dataframe[['SUM_OF_PROFIT']])
    dataframe["PROFIT"] = rs.fit_transform(dataframe[["PROFIT"]])
    dataframe["TYEAR*TESTABLISMENT*NEMPL"] = rs.fit_transform(dataframe[["TYEAR*TESTABLISMENT*NEMPL"]])
    dataframe["TYEAR+TESTABLISMENT+NEMPL"] = rs.fit_transform(dataframe[["TYEAR+TESTABLISMENT+NEMPL"]])
    dataframe["TETABLISEMENT+NEMPLOYEUR"] = rs.fit_transform(dataframe[["TETABLISEMENT+NEMPLOYEUR"]])
    dataframe["TETABLISEMENT*NEMPLOYEUR"] = rs.fit_transform(dataframe[["TETABLISEMENT*NEMPLOYEUR"]])
    dataframe["ESTABLISHMENT_TENURE"] = rs.fit_transform(dataframe[["ESTABLISHMENT_TENURE"]])
    dataframe["TOTAL_RISK"] = rs.fit_transform(dataframe[["TOTAL_RISK"]])
    dataframe["NUMBER_OF_EMP"] = rs.fit_transform(dataframe[["NUMBER_OF_EMP"]])
    dataframe["TOTAL_LIMIT"] = rs.fit_transform(dataframe[["TOTAL_LIMIT"]])
    dataframe["TOTAL_RISK_RATE"] = rs.fit_transform(dataframe[['TOTAL_RISK_RATE']])

    return dataframe

# ohe_df = robust_scaler(ohe_df)
# ohe_df.head()

###################################################
# SPLIT INTO TEST AND TRAIN
###################################################
def model_prep(dataframe,test_size=0.50):
    dataframe = dataframe.drop(['CUSTOMER_NUM'], axis=1)
    y = dataframe['SALES']
    X = dataframe.drop(['SALES'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=45,stratify=y)
    return dataframe,X,y,X_train,y_train,X_test,y_test

# ohe_df,X,y,X_train,y_train,X_test,y_test=model_prep(ohe_df)



######################################################
# Base Models skeatlearn kutuphanesindekilere gore
######################################################
def base_models(X, y, X_train, y_train, X_test, y_test,skf = StratifiedKFold(n_splits=10)):
    models = [('LR', LogisticRegression().fit(X_train, y_train)),
              ('KNN', KNeighborsClassifier().fit(X_train, y_train)),
              ("SVC", SVC(random_state=1).fit(X_train, y_train)),
              ("CART", DecisionTreeClassifier(random_state=1).fit(X_train, y_train)),
              ("RF", RandomForestClassifier(random_state=1).fit(X_train, y_train)),
              ('Adaboost', AdaBoostClassifier(random_state=1).fit(X_train, y_train)),
              ('GBM', GradientBoostingClassifier(random_state=1).fit(X_train, y_train)),
              ('LightGBM', LGBMClassifier(random_state=1).fit(X_train, y_train)),
              ('CatBoost', CatBoostClassifier(random_state=1, verbose=False).fit(X_train, y_train)),
              ('XGBoost', XGBClassifier(random_state=1).fit(X_train, y_train))
              ]

    #https://medium.com/@g.canguven11/dengesi̇z-veri̇-kümeleri̇-i̇le-maki̇ne-öğrenmesi̇-63bbac5f6869
    for name, classifier in models:
        cv_results = cross_val_score(classifier, X_test, y_test, cv=skf.split(X_test,y_test),scoring='roc_auc')
        print(f"roc_auc: {round(cv_results.mean(), 4)} ({name}) ")
    return
# skf = StratifiedKFold(n_splits=10) icin cikan sonuclar
# roc_auc: 0.761 (LR)
# roc_auc: 0.7221 (KNN)
# roc_auc: 0.7493 (SVC)
# roc_auc: 0.5985 (CART)
# roc_auc: 0.8123 (RF)
# roc_auc: 0.7752 (Adaboost)
# roc_auc: 0.7989 (GBM)
# roc_auc: 0.8006 (LightGBM)
# roc_auc: 0.8173 (CatBoost)
# roc_auc: 0.7995 (XGBoost)
# base_models()

""" 
################################
# Analyzing Model Complexity with Learning Curves
################################
not:  bu grafikler hiperparametre seciminde kullanildi.
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20,40]],
                 ["n_estimators", [10, 50, 100, 200, 500,750]]]
gbm_val_params = [["max_depth", [5, 8, 15, 20, None]],
                 ['min_samples_leaf',[2,4,6,10]],
                 ["max_features", [3,7,15,24,30]],
                 ["min_samples_split", [2, 5, 8, 10]],
                 ["n_estimators", [100,500,1000,15000]],
                ['subsample',[0.5,0.7,1]]]
xgb_val_params=[["max_depth",[5,10,15,20]],
                ['gamma',[0.1,0.05,0.03,0.01]],
                ['min_child_weight',[0.1,0.2,0.5,1]],
                ['max_delta_step',[2,5,10]],
                ['subsample',[0.5,0.8,1]],
                ['colsample_bytree',[0.3,0.5,0.7,1]]]
lgb_val_params=[['num_iterations',[100,1000,10000]],
                ['bagging_fraction',[0.5,0.7,1]],
                ["max_depth",[5,10,13,20]],
                ['num_leaves',[2,5,10,15,20]],
                ['bagging_freq',[3,5,7]],
                ['min_data_in_leaf',[2,4,6]],
                ['max_bin',[20,30,34,40]],
                ['feature_fraction',[0.5,0.7,1]]]

catboost_val_params=[["iterations",[100,1000,2000,2500]],
                     ["depth",[2,5,7,10]]]

rf_model = RandomForestClassifier(random_state=17)
gbm_model = GradientBoostingClassifier(random_state=1,verbose=False)
lgb_model=lgb.LGBMClassifier(random_state=1)
xgb_model=xgb.XGBClassifier(random_state=1)
catboost_model=CatBoostClassifier(random_state=1,verbose=False)

def model_curve(model, X, y, params, scoring="roc_auc", cv=10):
    for i in range(len(params)):
        val_curve_params(model, X, y, params[i][0], params[i][1])
        print(gbm_val_params[i][0])
        print(gbm_val_params[i][1])
model_curve(rf_model,X,y,rf_val_params,cv=3)
model_curve(gbm_model,X,y,gbm_val_params,cv=3)
model_curve(lgb_model,X,y,lgb_val_params,cv=3)
model_curve(xgb_model,X,y,xgb_val_params,cv=3)
model_curve(catboost_model,X,y,catboost_val_params,cv=3)

"""
############################################################################################################
# Random Forests
############################################################################################################
def RandomForest(X, y, X_train, y_train, X_test, y_test,skf = StratifiedKFold(n_splits=10),scoring="roc_auc",roc_curve=False,random_state=45):
    print('#########################  RF   ##############################')
    rf_params = {"max_depth": 19,
                 "max_features": 3,
                "min_samples_split": 10,
                "n_estimators": 240}
    rf_tuned = RandomForestClassifier(**rf_params, random_state=random_state).fit(X_train, y_train)
    # cross_validate
    tuned_cv_results = cross_validate(rf_tuned, X, y, cv=10, scoring=scoring,n_jobs=-1)
    print('KFold cross validate')
    print(f"roc_auc_mean :{tuned_cv_results['test_score'].mean()}")
    # cross_val_score
    cv_results = cross_val_score(rf_tuned, X_test, y_test, cv=skf.split(X_test, y_test), scoring=scoring,n_jobs=-1)
    print('Stratifield Kfold cross val score')
    print(f"roc_auc_mean:{cv_results.mean()}")
    # ROC Curve
    if roc_curve:
        plot_roc_curve(rf_tuned, X_test, y_test)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.show()
    # plot_confusion_matrix
    y_pred = rf_tuned.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y_test')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()
    return rf_tuned
# rf_tuned = RandomForest(roc_curve=True)
#########################  RF   ##############################
# KFold cross validate
# roc_auc_mean :0.7867396179220114
# Stratifield Nfold cross val score
# roc_auc_mean:0.8110066161054071

############################################################################################################
#GBM
############################################################################################################
def GradientBoosting(X, y, X_train, y_train, X_test, y_test,skf = StratifiedKFold(n_splits=10),scoring="roc_auc",roc_curve=False,random_state=45):
    print('#########################  GBM   ##############################')
    gbm_params = {"learning_rate": 0.02, 'max_depth': 6,
                   'max_features': 24,
                   'min_samples_leaf': 2,
                   'min_samples_split': 2,
                   'n_estimators': 1500,
                   'subsample': 0.7,
                   'verbose': False}
    gbm_tuned = GradientBoostingClassifier(**gbm_params, random_state=random_state).fit(X_train, y_train)
    # cross_validate
    tuned_cv_results = cross_validate(gbm_tuned, X, y, cv=10, scoring=scoring,n_jobs=-1)
    print('KFold cross validate')
    print(f"roc_auc_mean :{tuned_cv_results['test_score'].mean()}")
    # cross_val_score
    cv_results = cross_val_score(gbm_tuned, X_test, y_test, cv=skf.split(X_test, y_test), scoring=scoring,n_jobs=-1)
    print('Stratifield Nfold cross val score')
    print(f"roc_auc_mean:{cv_results.mean()}")
    # ROC Curve
    if roc_curve:
        plot_roc_curve(gbm_tuned, X_test, y_test)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.show()
    # plot_confusion_matrix
    y_pred = gbm_tuned.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y_test')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()
    return gbm_tuned
# gbm_tuned=GradientBoosting(roc_curve=True)


#########################  GBM   ##############################
# KFold cross validate
# roc_auc_mean :0.7849637975048631
# Stratifield Nfold cross val score
# roc_auc_mean:0.8185265576207115

#The cross_validate function differs from cross_val_score in two ways -
#It allows specifying multiple metrics for evaluation.
#It returns a dict containing training scores, fit-times and score-times in addition to the test score.

############################################################################################################
# XGBoost
############################################################################################################
def XGBoost(X, y, X_train, y_train, X_test, y_test,skf = StratifiedKFold(n_splits=10),scoring="roc_auc",roc_curve=False,random_state=45):
    print('#########################  XGBOOST   ##############################')
    xgb_params = {'objective': 'binary:logistic',
                  'eval_metric': "auc",
                  'seed': 1,'max_depth': 5,
                  'eta':0.01,
                  'gamma':0.01,
                  'min_child_weight': 0.2,
                  'max_delta_step': 10,
                  'subsample':0.8,
                  'colsample_bytree':0.5,
                  'sampling_method':'uniform'}
    # auc train and test
    df_train = xgb.DMatrix(X_train, y_train)
    df_test = xgb.DMatrix(X_test, y_test)
    watchlist = [(df_train, 'train'), (df_test, 'valid')]
    num_round=6800
    xgb_base_model = xgb.train(xgb_params, df_train,
                           num_round,watchlist,
                           verbose_eval=200,early_stopping_rounds=200)

    #K-fold Cross validation
    df_xgb=xgb.DMatrix(X,y)
    xgb_cv = xgb.cv(xgb_params, df_xgb,
                num_boost_round=10000,
                early_stopping_rounds=200, nfold=10,
                verbose_eval=500)
    print('KFold cross validate')
    print(f"roc_auc_mean :{np.mean(xgb_cv)}")
    #Cross_val_score
    xgb_tuned = xgb.XGBClassifier(**xgb_params,n_estimators=6800,use_label_encoder=False)
    xgb_tuned.fit(X_train,y_train)
    cv_results = cross_val_score(xgb_tuned, X_test, y_test, cv=skf.split(X_test, y_test),n_jobs=-1, scoring='roc_auc')
    print('Stratifield Nfold cross val score')
    print(f"roc_auc_mean:{cv_results.mean()}")
    # ROC Curve
    if roc_curve:
        plot_roc_curve(xgb_tuned, X_test, y_test)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.show()
    # plot_confusion_matrix
    y_pred = xgb_tuned.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y_test')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()
    return xgb_tuned
# xgb_tuned = XGBoost(roc_curve=True)

# #########################  XGBOOST   ##############################
# [0]	train-auc:0.80057	valid-auc:0.72686
# [200]	train-auc:0.89376	valid-auc:0.79471
# [400]	train-auc:0.93092	valid-auc:0.80437
# [600]	train-auc:0.95651	valid-auc:0.81089
# [800]	train-auc:0.97284	valid-auc:0.81505
# [1000]	train-auc:0.98364	valid-auc:0.81788
# [1200]	train-auc:0.99028	valid-auc:0.82066
# [1400]	train-auc:0.99437	valid-auc:0.82168
# [1600]	train-auc:0.99679	valid-auc:0.82299
# [1800]	train-auc:0.99816	valid-auc:0.82356
# [2000]	train-auc:0.99913	valid-auc:0.82402
# [2200]	train-auc:0.99958	valid-auc:0.82450
# [2400]	train-auc:0.99981	valid-auc:0.82506
# [2496]	train-auc:0.99987	valid-auc:0.82485
# KFold cross validate
# roc_auc_mean :train-auc-mean   0.972
# train-auc-std    0.001
# test-auc-mean    0.845
# test-auc-std     0.012
# dtype: float64
# Stratifield Nfold cross val score
# roc_auc_mean:0.8100946031251185
############################################################################################################
# LightGBM
############################################################################################################
def LightGBM(X, y, X_train, y_train, X_test, y_test,skf = StratifiedKFold(n_splits=10),scoring="roc_auc",roc_curve=False,random_state=45):
    print('#########################  LIGHTGBM   ##############################')
    lgb_params = {'objective': 'binary',
                  'metric': "AUC", 'seed': 1,
                  'num_iterations': 1200,
                  'boosting': 'gbdt',
                  "learning_rate": 0.03,
                  'bagging_fraction': 0.7,
                  "max_depth": 11,
                  'num_leaves': 15,
                  'bagging_freq': 3,
                  'min_data_in_leaf': 2,
                  'max_bin': 34,
                  'feature_fraction': 0.7}
    # auc train and test
    df_train = lgb.Dataset(X_train, y_train)
    df_test = lgb.Dataset(X_test, y_test, reference=df_train)
    lgb.train(lgb_params, df_train,
              valid_sets=[df_train, df_test],
              verbose_eval=100)

    #K-fold Cross validation
    df_lgb = lgb.Dataset(X, y)
    lgb_cv = lgb.cv(lgb_params, df_lgb, nfold=10, verbose_eval=False)


    #Cross_val_score
    lgb_tuned = lgb.LGBMClassifier(**lgb_params, verbose=500).fit(X_train, y_train)
    cv_results = cross_val_score(lgb_tuned, X_test, y_test, cv=skf.split(X_test, y_test),n_jobs=-1, scoring='roc_auc',verbose=False)
    print('KFold cross validate')
    print(f"roc_auc_mean :{np.mean(lgb_cv['auc-mean'])}")
    print('Stratifield Nfold cross val score')
    print(f"roc_auc_mean:{cv_results.mean()}")
    # ROC Curve
    if roc_curve:
        plot_roc_curve(lgb_tuned, X_test, y_test)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.show()
    # plot_confusion_matrix
    y_pred = lgb_tuned.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y_test')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()
    return lgb_tuned
# lgb_tuned = LightGBM(roc_curve=True)


# '#########################  LIGHTGBM   ##############################'
# [1200]	training's auc: 0.999991	valid_1's auc: 0.829049
# KFold cross validate
# roc_auc_mean :0.8399374672786619
# Stratifield Nfold cross val score
# roc_auc_mean:0.8201049096780373

############################################################################################################
# CatBoostClassifier
############################################################################################################

def CatBoost(X, y, X_train, y_train, X_test, y_test,skf = StratifiedKFold(n_splits=10),scoring="roc_auc",random_state=45,roc_curve=False):
    print('#########################  CATBOOST   ##############################')
    catboost_params = {"iterations": 2500,
                       "learning_rate": 0.01,
                        "depth": 7}
    catboost_tuned = CatBoostClassifier(**catboost_params, random_state=1, verbose=False).fit(X_train,y_train)
    # cross_validate
    tuned_cv_results = cross_validate(catboost_tuned, X, y, cv=10, scoring=scoring,n_jobs=-1)
    print('KFold cross validate')
    print(f"roc_auc_mean :{tuned_cv_results['test_score'].mean()}")
    # cross_val_score
    cv_results = cross_val_score(catboost_tuned, X_test, y_test, cv=skf.split(X_test, y_test), scoring=scoring,n_jobs=-1)
    print('Stratifield Nfold cross val score')
    print(f"roc_auc_mean:{cv_results.mean()}")
    # ROC Curve
    if roc_curve:
        plot_roc_curve(catboost_tuned, X_test, y_test)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.show()
    # plot_confusion_matrix
    y_pred = catboost_tuned.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y_test')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()
    return catboost_tuned
# catboost_tuned= CatBoost(roc_curve=True)

#########################  CATBOOST   ##############################
# KFold cross validate
# roc_auc_mean :0.7954677408240597
# Stratifield Nfold cross val score
# roc_auc_mean:0.8190237259794351

############################################################################################################
# Stacking & Ensemble Learning
############################################################################################################

def voting(lgb_tuned,xgb_tuned,X_train, y_train, X_test, y_test):
    print("#####################   STACKING & ENSEMBLE LEARNING   ##############################")
    voting_clf = VotingClassifier(estimators=[('LGB', lgb_tuned),('XGB', xgb_tuned)], voting='soft')
    voting_clf.fit(X_train, y_train)
    cv_results = cross_validate(voting_clf, X_test, y_test, cv=10, scoring="roc_auc")
    print(f"voting_auc_mean:{cv_results['test_score'].mean()}")
# voting()

# voting_auc_mean:0.8198660912667919

#########################################
# KORELASYON INCELEMESI VE YUKSEK KORELASYONLULARIN CIKARILMASI
def correlationmatrix(dataframe):
    f, ax = plt.subplots(figsize=[25, 20])
    sns.heatmap(dataframe.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
    ax.set_title("Correlation Matrix", fontsize=20)
    plt.show()
    #korelasyon matris ciktisi
    dataframe.columns
    high_corr = dataframe.corr()
    for i in high_corr.columns:
        print(high_corr[i].sort_values(ascending=False)[1:6])
        print('##################################################')
# correlationmatrix(ohe_df)

###################

###### FEATURE IMPORTANCE'LARIN ORTALAMALARININ TREE BASED SELECTION YONTEMINE GORE BULUNMASI ###################

# def plot_importance(model, features, num=len(X), save=False):
#     feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
#     plt.figure(figsize=(14, 14))
#     sns.set(font_scale=1)
#     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
#                                                                      ascending=False)[0:num])
#     plt.title('Features')
#     plt.tight_layout()
#     plt.show()
#     if save:
#         plt.savefig(f"{model}_feature_importances.png")
#
#     return feature_imp.sort_values(by="Value", ascending=False)
# def average_feature_importance():
#     df_xgb=plot_importance(xgb_tuned,X)
#     df_lgb = plot_importance(lgb_tuned, X)
#     df_catboost = plot_importance(catboost_tuned, X)
#     df_gbm = plot_importance(gbm_tuned, X)
#     df_rf=plot_importance(rf_tuned,X)
#
#     ### OLUSAN value degerlerini ayni seviyeye getirmek icin 0 1 arasinda scale ediyoruz.
#     scaler = MinMaxScaler((0, 1))
#     df_lgb['Value'] = scaler.fit_transform(df_lgb[['Value']])
#     df_gbm['Value'] = scaler.fit_transform(df_gbm[['Value']])
#     df_catboost['Value'] = scaler.fit_transform(df_catboost[['Value']])
#     df_rf['Value']=scaler.fit_transform(df_rf[['Value']])
#     df_xgb['Value']=scaler.fit_transform(df_xgb[['Value']])
#     # model adlarinida df lere colon olarak ekleyelim.
#     df_lgb['model_name'] = 'lgb'
#     df_gbm['model_name'] = 'gbm'
#     df_catboost['model_name'] = 'catboost'
#     df_rf['model_name']='rf'
#     df_xgb['model_name']='xgb'
#
#     # 5 modele gore olusan feature importance lari concat yapip bir df olusturalim.
#     df_importance = pd.concat([df_lgb, df_gbm, df_catboost,df_rf,df_xgb])
#     # value adli kolonun ismini feature importance yapiyoruz.
#     df_importance['feature_importance'] = df_importance['Value']
#     df_importance = df_importance.drop('Value', axis=1)
#     # feature lara gore groupby alip feature importance in ortalamasini bulalim.
#     df_importance_final = df_importance.groupby('Feature').agg({'feature_importance': 'mean'}).sort_values(
#         'feature_importance',ascending=False)
#     return df_importance_final
# df_importance_final=average_feature_importance()

# df_importance_final
# Out[22]:
#                                          feature_importance
# Feature
# NEW_ESTABLISHMENT_TENURE_61-119                       0.000
# CAT_NUMBER_OF_EMP_201-3333                            0.001
# NEW_YEAR_SECTOR_MIDDLE_SERVICES                       0.036
# NEW_YEAR_PROFIT_YOUNG_LOW_PROFIT                      0.041
# NEW_YEAR_PROFIT_MIDDLE_SMALL_PROFIT                   0.049
# NEW_YEAR_SECTOR_MATURE_OTHERS                         0.049
# NEW_ESTABLISHMENT_TENURE_41-60                        0.050
# NEW_ESTABLISHMENT_TENURE_21-40                        0.053
# NEW_YEAR_SECTOR_YOUNG_OTHERS                          0.055
# CUS_TENURE_2                                          0.056
# CUS_TENURE_3                                          0.056
# NEW_YEAR_SECTOR_YOUNG_SERVICES                        0.057
# NEW_YEAR_SECTOR_MIDDLE_RETAIL-WHOLESALE               0.057
# NEW_YEAR_SECTOR_MIDDLE_OTHERS                         0.058
# NEW_YEAR_PROFIT_MIDDLE_NORMAL_PROFIT                  0.059
# NEW_YEAR_SECTOR_MATURE_SERVICES                       0.059
# REGION_Southeastern Anatolia Region                   0.059
# NEW_YEAR_TENURE_MIDDLE                                0.060
# NEW_YEAR_PROFIT_YOUNG_SMALL_PROFIT                    0.063
# NEW_YEAR_SECTOR_YOUNG_MANUFACTURING                   0.064
# YEAR_TENURE_5                                         0.064
# SECTOR_SERVICES                                       0.064
# CAT_NUMBER_OF_EMP_16-200                              0.065
# YEAR_TENURE_6                                         0.067
# NEW_YEAR_SECTOR_YOUNG_RETAIL-WHOLESALE                0.068
# CUS_TENURE_6                                          0.069
# REGION_Eastern Anatolia Region                        0.070
# YEAR_TENURE_2                                         0.070
# CUS_TENURE_4                                          0.070
# LAST_CREDIT_TIME_8                                    0.071
# NEW_YEAR_SECTOR_MIDDLE_MANUFACTURING                  0.072
# YEAR_TENURE_7                                         0.073
# LAST_CREDIT_TIME_3                                    0.074
# NUMBER_OF_TRANSACTIONS_8                              0.076
# LAST_CREDIT_TIME_6                                    0.076
# NEW_YEAR_PROFIT_MATURE_SMALL_PROFIT                   0.076
# CUS_TENURE_5                                          0.076
# NUMBER_OF_TRANSACTIONS_5                              0.078
# NEW_YEAR_TENURE_YOUNG                                 0.081
# REGION_Black Sea Region                               0.083
# CUS_TENURE_8                                          0.084
# REGION_Mediterranean Region                           0.084
# NUMBER_OF_TRANSACTIONS_7                              0.085
# YEAR_TENURE_3                                         0.087
# LAST_CREDIT_TIME_7                                    0.089
# NEW_ESTABLISHMENT_TENURE_11-14                        0.090
# NEW_ESTABLISHMENT_TENURE_15-20                        0.090
# REGION_Marmara Region                                 0.091
# LAST_CREDIT_TIME_4                                    0.092
# NEW_PROFIT_MIDDLE                                     0.094
# CUS_TENURE_7                                          0.094
# YEAR_TENURE_4                                         0.094
# NUMBER_OF_TRANSACTIONS_2                              0.098
# NUMBER_OF_TRANSACTIONS_3                              0.100
# CAT_NUMBER_OF_EMP_5_7                                 0.105
# CAT_NUMBER_OF_EMP_8-15                                0.106
# NEW_YEAR_SECTOR_MATURE_RETAIL-WHOLESALE               0.107
# NEW_PROFIT_SMALL                                      0.110
# LAST_CREDIT_TIME_5                                    0.111
# NUMBER_OF_TRANSACTIONS_4                              0.111
# SECTOR_OTHERS                                         0.112
# LAST_CREDIT_TIME_2                                    0.114
# SECTOR_RETAIL-WHOLESALE                               0.121
# REGION_Central Anatolia Region                        0.122
# YEAR_TENURE_8                                         0.151
# NUMBER_OF_TRANSACTIONS_6                              0.158
# NEW_PROFIT_VERY_SMALL                                 0.294
# TYEAR+TESTABLISMENT+NEMPL                             0.369
# TETABLISEMENT+NEMPLOYEUR                              0.393
# TETABLISEMENT*NEMPLOYEUR                              0.419
# NUMBER_OF_EMP                                         0.421
# TYEAR*TESTABLISMENT*NEMPL                             0.421
# TOTAL_RISK_RATE                                       0.435
# ESTABLISHMENT_TENURE                                  0.464
# SUM_OF_RISK                                           0.528
# SUM_OF_LIMIT                                          0.561
# SUM_OF_PROFIT                                         0.564
# TOTAL_RISK                                            0.575
# PROFIT                                                0.749
# TOTAL_LIMIT                                           0.862

###korelasyon ve feature importance lara bakarak yuksek korelasyonlu onemli featurelarin cikarilmasi
def drop(dataframe):
    print("###################   FEATURE SELECTION    ##############################")
    print("Yuksek korelasyonlulara ve feature importancelarin ortalamasina bakarak bazi featurelar elendi.")
    ohe_df = dataframe.drop(['NEW_YEAR_TENURE_YOUNG',
                          'NEW_YEAR_PROFIT_MIDDLE_SMALL_PROFIT',
                          'NEW_ESTABLISHMENT_TENURE_21-40',
                          'SUM_OF_RISK', 'TOTAL_LIMIT',
                          'TYEAR+TESTABLISMENT+NEMPL',
                          'TETABLISEMENT+NEMPLOYEUR',
                          'NEW_ESTABLISHMENT_TENURE_61-119',
                          'NEW_YEAR_SECTOR_MIDDLE_SERVICES','PROFIT',
                          'NUMBER_OF_EMP','NEW_ESTABLISHMENT_TENURE_41-60',
                          'NEW_YEAR_PROFIT_YOUNG_LOW_PROFIT', 'NEW_YEAR_SECTOR_YOUNG_SERVICES',
                          'CUS_TENURE_3','CAT_NUMBER_OF_EMP_201-3333','CUS_TENURE_2',
                          'NEW_YEAR_SECTOR_MATURE_OTHERS',
                          'NUMBER_OF_TRANSACTIONS_8',
                          'YEAR_TENURE_7', 'LAST_CREDIT_TIME_8',
                          'YEAR_TENURE_2','YEAR_TENURE_6', 'YEAR_TENURE_5',
                          'NEW_YEAR_SECTOR_YOUNG_MANUFACTURING','YEAR_TENURE_3',
                          'NEW_YEAR_SECTOR_MIDDLE_OTHERS',
                          'CUS_TENURE_5',
                          'NEW_YEAR_SECTOR_YOUNG_RETAIL-WHOLESALE',
                          'NEW_YEAR_SECTOR_MIDDLE_MANUFACTURING',
                          'NUMBER_OF_TRANSACTIONS_7',
                          'NEW_YEAR_PROFIT_MIDDLE_NORMAL_PROFIT',
                          'CAT_NUMBER_OF_EMP_16-200',
                          'LAST_CREDIT_TIME_3','YEAR_TENURE_4',
                          'LAST_CREDIT_TIME_6',
                          'NEW_YEAR_TENURE_MIDDLE',
                          'NEW_YEAR_PROFIT_YOUNG_SMALL_PROFIT',
                          'NUMBER_OF_TRANSACTIONS_5',
                          'NEW_YEAR_PROFIT_MATURE_SMALL_PROFIT',
                          'CUS_TENURE_4',
                          'NEW_PROFIT_MIDDLE',
                          'CAT_NUMBER_OF_EMP_8-15',
                          'NEW_YEAR_SECTOR_MATURE_SERVICES',
                          'SECTOR_SERVICES',
                          'NEW_YEAR_SECTOR_YOUNG_OTHERS',
                          'CUS_TENURE_6',
                          'NEW_ESTABLISHMENT_TENURE_11-14',
                          'LAST_CREDIT_TIME_7',
                          'NEW_ESTABLISHMENT_TENURE_15-20'], axis=1)
    return ohe_df

# TETABLISEMENT+NEMPLOYEUR    0.995
# TYEAR+TESTABLISMENT+NEMPL   0.993
# TETABLISEMENT*NEMPLOYEUR    0.938
# TYEAR*TESTABLISMENT*NEMPL   0.923
# LAST_CREDIT_TIME_8          0.081
# Name: NUMBER_OF_EMP, dtype: float64
# ##################################################
# TOTAL_LIMIT                      0.890
# SUM_OF_RISK                      0.760
# SUM_OF_LIMIT                     0.573
# SALES                            0.094
# NEW_YEAR_SECTOR_YOUNG_SERVICES   0.064
# Name: TOTAL_RISK, dtype: float64

# ##################################################
# SUM_OF_PROFIT                          0.647
# NEW_PROFIT_MIDDLE                      0.302
# NEW_YEAR_PROFIT_YOUNG_LOW_PROFIT       0.221
# NEW_YEAR_PROFIT_MIDDLE_NORMAL_PROFIT   0.205
# NEW_PROFIT_SMALL                       0.172
# Name: PROFIT, dtype: float64
# ##################################################
# TOTAL_RISK                       0.890
# SUM_OF_LIMIT                     0.724
# SUM_OF_RISK                      0.687
# SALES                            0.146
# NEW_YEAR_SECTOR_YOUNG_SERVICES   0.064
# Name: TOTAL_LIMIT, dtype: float64
# ##################################################
# ohe_df=drop(ohe_df)
######################################
# korelasyonlu degerler cikartilarak butun modellere gore tekrar final modeller kuralim.

######################################
# MODELLEME 2
# ohe_df,X,y,X_train,y_train,X_test,y_test=model_prep(ohe_df)
# check_df(ohe_df)
######################################################
# Base Models gore(feature selection yapilarak)
######################################################
# base_models()
# roc_auc: 0.7579 (LR)
# roc_auc: 0.7087 (KNN)
# roc_auc: 0.7272 (SVC)
# roc_auc: 0.6126 (CART)
# roc_auc: 0.8022 (RF)
# roc_auc: 0.7664 (Adaboost)
# roc_auc: 0.7832 (GBM)
# roc_auc: 0.7923 (LightGBM)
# roc_auc: 0.8024 (CatBoost)
# roc_auc: 0.7902 (XGBoost)
############################################################################################################
# Random Forests
############################################################################################################
# rf_tuned = RandomForest(roc_curve=True)
#########################  RF   ##############################
# KFold cross validate
# roc_auc_mean :0.7683454622688424
# Stratifield Nfold cross val score
# roc_auc_mean:0.7972859645814416

############################################################################################################
#GBM
############################################################################################################
# gbm_tuned=GradientBoosting(roc_curve=True)

# #########################  GBM   ##############################
# KFold cross validate
# roc_auc_mean :0.7657863404071507
# Stratifield Nfold cross val score
# roc_auc_mean:0.804952585042195
############################################################################################################
# XGBoost
############################################################################################################
# xgb_tuned=XGBoost(roc_curve=True)

#########################  XGBOOST   ##############################
# [0]	train-auc:0.70950	valid-auc:0.66591
# [200]	train-auc:0.88132	valid-auc:0.77902
# [400]	train-auc:0.91654	valid-auc:0.78748
# [600]	train-auc:0.94372	valid-auc:0.79335
# [800]	train-auc:0.96205	valid-auc:0.79788
# [1000]	train-auc:0.97495	valid-auc:0.80071
# [1200]	train-auc:0.98285	valid-auc:0.80273
# [1400]	train-auc:0.98844	valid-auc:0.80418
# [1600]	train-auc:0.99214	valid-auc:0.80485
# [1800]	train-auc:0.99482	valid-auc:0.80525
# [1844]	train-auc:0.99522	valid-auc:0.80531
# KFold cross validate
# roc_auc_mean :train-auc-mean   0.962
# train-auc-std    0.001
# test-auc-mean    0.829
# test-auc-std     0.016
# dtype: float64
# Stratifield Nfold cross val score
# roc_auc_mean:0.8004123898945791
############################################################################################################
# LightGBM
############################################################################################################
# lgb_tuned=LightGBM(roc_curve=True)

# [1200]	training's auc: 0.999726	valid_1's auc: 0.804455
# KFold cross validate
# roc_auc_mean :0.8231645470248538
# Stratifield Nfold cross val score
# roc_auc_mean:0.7962825931848518

############################################################################################################
# CatBoostClassifier
############################################################################################################
# catboost_tuned=CatBoost(roc_curve=True)

# #########################  CATBOOST   ##############################
# KFold cross validate
# roc_auc_mean :0.7743598534305984
# Stratifield Nfold cross val score
# roc_auc_mean:0.803313898993026

############################################################################################################
# Stacking & Ensemble Learning
############################################################################################################
# voting(lgb_tuned,xgb_tuned,X_train, y_train, X_test, y_test)

# voting_auc_mean:0.8084360613540961


############################################################################################################
# PIPELINE
############################################################################################################

def pipeline_prediction_sales():
    print("1")
    df = pd.read_csv("datasets/Final_Project.csv")
    print("2")
    df, df_3 = data_prep(df)
    print("3")
    df = missing_value(df)
    print("4")
    df=feature_engineering(df)
    check_df(df)
    print("5")
    ohe_df = one_hot(df)
    print("6")
    ohe_df = robust_scaler(ohe_df)
    print("7")
    ohe_df,X, y, X_train, y_train, X_test, y_test = model_prep(ohe_df)
    print("8")
    base_models(X, y, X_train, y_train, X_test, y_test)
    print("9")
    print("############ HIPERPARAMETRE OPTIMIZASYONLARI YAPILDI ############")
    rf_tuned = RandomForest(X, y, X_train, y_train, X_test, y_test,roc_curve=True)
    print("10")
    gbm_tuned = GradientBoosting(X, y, X_train, y_train, X_test, y_test,roc_curve=True)
    print("11")
    xgb_tuned = XGBoost(X, y, X_train, y_train, X_test, y_test,roc_curve=True)
    print("12")
    lgb_tuned = LightGBM(X, y, X_train, y_train, X_test, y_test,roc_curve=True)
    print("13")
    catboost_tuned = CatBoost(X, y, X_train, y_train, X_test, y_test,roc_curve=True)
    print("14")
    voting(lgb_tuned,xgb_tuned,X_train, y_train, X_test, y_test)
    print("15")
    print("##  EN YUKSEK SKORLU XGBOOST U FINAL MODEL OLARAK SECIYORUZ ##")
    final_model=xgb_tuned

    return X,y,ohe_df,final_model
X,y,ohe_df,final_model= pipeline_prediction_sales()
###########################################################################################################
# PICKLE 'A CEVIRME
###########################################################################################################
import os
import joblib
import pickle
os.chdir("./")
joblib.dump(final_model, "final_proje_predict.pkl")
finalmodel = joblib.load("./final_proje_predict.pkl")

######################################################
# Prediction for a New Observation
######################################################
random_user = X.sample(1)
finalmodel.predict(random_user)
ohe_df[ohe_df.index==random_user.index[0]]['SALES']

