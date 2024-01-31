import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import seaborn as sns
import calendar
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#Veri Setini Yükle
df = pd.read_csv("Dataset/LAX_flight_on_time_performance.csv")

#Veri Setinin Preprocessing Aşaması
df.isnull().sum()
df.drop(df[df["CANCELLED"] == 1].index, axis=0, inplace=True)

    #Gereksiz Sütunların Çıkarılması
unnecessary_cols = ["YEAR", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_CITY_NAME", "ORIGIN_WAC", "DEST_WAC",
                    "CANCELLED", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]

df.drop(columns=unnecessary_cols, axis=1, inplace=True)

    #Air_time boş kayıtların düşürülmesi

df.drop(df[df["AIR_TIME"].isnull()].index, axis=0, inplace=True)



#Feature Engineering

    #Haftaiçi vs Hafta sonunu belirten yeni sütun
df["n_PART_OF_WEEK"] = df["DAY_OF_WEEK"].map(lambda x: "weekday" if x <= 5 else "Weekend")


    #Saat bilgisini veren CRS_DEP_TIME değişkenini düzeltme
df["CRS_DEP_TIME"] = df["CRS_DEP_TIME"].astype(str)

def fix_hour(df):
    # Saat verisini doğru biçime getir
    if len(df) == 2:
        return f"00:{df}"
    elif len(df) == 3:
        return f"0{df[0]}:{df[-2:]}"
    elif len(df) == 4:
        return f"{df[:2]}:{df[-2:]}"
    else:
        return df

df["CRS_DEP_TIME"] = df["CRS_DEP_TIME"].apply(fix_hour)


    #Saat Kategorisi Oluşturma
def hour_cat(df):
    return df[:2]

df["TIME_CATEGORY"] = df["CRS_DEP_TIME"].apply(hour_cat)


    #Günün saatlik dilimleri= sabah, öğlen, akşam
df["TIME_CATEGORY"] = df["TIME_CATEGORY"].astype("int64")
def sep_day_time(saat):
    if saat >=0 and saat <6:
        return "night"
    if saat >= 6 and saat < 12:
        return "morning"
    if saat >= 12 and saat < 14:
        return "noon"
    if saat >= 14 and saat < 18:
        return "afternoon"
    else:
        return "evening"

df["n_PART_OF_DAY"] = df["TIME_CATEGORY"].apply(sep_day_time)


    #Mevsim Kategorileri Oluşturma
def categorize_season(month):
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Fall'
    else:
        return 'Winter'

df["new_SEASON"] = df["MONTH"].apply(categorize_season)


    #Ayları stringe çevirme

df["MONTH"] = df["MONTH"].apply(lambda x: calendar.month_name[x])


    #Günleri dönüştür
days_numeric = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}

df["DAY_OF_WEEK"] = df["DAY_OF_WEEK"].map(days_numeric)



    #Distance kategorik değişkeni
df["NEW_DISTANCE_CAT"] = pd.cut(x=df["DISTANCE"], bins=[88, 380, 1000, 1700, 2616], labels=["short_range", "over_state", "long_range", "overseas"])

df["DEP_DELAY_NEW"] = df["DEP_DELAY_NEW"].astype("int64")


#Outliers

a = df[df["DEP_DELAY_NEW"] > 0]

sns.boxplot(x="DEP_DELAY_NEW", data=a)
plt.show()


# def check_outlier(dataframe, col_name, q1=0, q3=0.95):
#     low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
#     if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
#         return True
#     else:
#         return False


def outlier_thresholds(dataframe, col_name, q1=0, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# check_outlier(df, "DEP_DELAY_NEW")
outlier_thresholds(df, "DEP_DELAY_NEW")
replace_with_thresholds(df, "DEP_DELAY_NEW")

df["DEP_DELAY_NEW"] = df["DEP_DELAY_NEW"].astype("int64")


#Label Encoding

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)


cat_cols = ['MONTH',
            "DAY_OF_MONTH",
            'DAY_OF_WEEK',
            'OP_UNIQUE_CARRIER',
            "DEST_CITY_NAME",
            'N_PART_OF_WEEK',
            'N_PART_OF_DAY',
            'NEW_SEASON',
            'NEW_DISTANCE_CAT']

num_cols = ["AIR_TIME", "DISTANCE"]


#One Hot Encoder
df.columns = [col.upper() for col in df.columns]

def one_hot_encoder(dataframe, cat_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)


    #Sayısal Değişkenleri Standartlaştırma

df = df.reset_index(drop=True)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#Model Oluşturma


    #MODEL

removes = ["ORIGIN", "DEST_AIRPORT_ID", "DEST", "CRS_DEP_TIME", "DEP_DELAY_NEW", "DEP_DEL15", "TIME_CATEGORY"]

X = df.drop(columns=removes, axis=1)
y = df["DEP_DELAY_NEW"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X.columns = [col.upper() for col in X.columns]

########################################################################################################################
#Lineer regresyon modelini oluşturun ve eğitin
model = LinearRegression()
model.fit(X_train, y_train)

#Test veri seti üzerinde tahmin yapın
y_pred = model.predict(X_test)

#Model performansını değerlendirin
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

sns.scatterplot(x=y_test, y=y_pred)
########################################################################################################################

X.columns = [col.replace(" ", "_") for col in X.columns]
X.columns = [col.replace(",", "_") for col in X.columns]

X_train.columns = [col.replace(" ", "_") for col in X.columns]
X_train.columns = [col.replace(",", "_") for col in X.columns]

models = [#('LR', LinearRegression()),
          #('KNN', KNeighborsRegressor()),
          # ("SVR", SVR()),
          #("CART", DecisionTreeRegressor()),
          #("RF", RandomForestRegressor()),
          # ('Adaboost', AdaBoostClassifier()),
          #('GBM', GradientBoostingRegressor()),
          #('XGBoost', XGBRegressor(objective="reg:squarederror")),
          ('LightGBM', LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    #RESULTS
# RMSE: 2321985863923.9126 (LR)
# RMSE: 30.5697 (KNN)
# RMSE: 38.8831 (CART)
# RMSE: 29.4488 (RF)
# RMSE: 28.6815 (GBM)
# RMSE: 28.9831 (XGBoost)
# RMSE: 28.3935 (LightGBM)

    #LGBM Parametre Optimizasyonu

# LightGBM modeli
model = LGBMRegressor()

param_grid = {
    "num_leaves": [50, 60, 70],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300]
}

# GridSearchCV

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# grid_search = RandomizedSearchCV(model, param_distributions=param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, n_iter=10)

grid_search.fit(X_train, y_train)

# En iyi parametreleri yazdırma
print("Best params:", grid_search.best_params_)


    #FINAL MODEL

params = {
    "num_leaves": 60,
    "learning_rate": 0.1,
    "n_estimators": 300
}

final_model = LGBMRegressor(**params)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


rmse = np.sqrt(mse)
print("RMSE:", rmse)

# FINAL MODEL RMSE: 28.36279148069317
