# Import các thư viện cần sử dụng

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import plotly.express as px
from sklearn import linear_model

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer  


import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


#_____________________________________Do Duy Long____________________________________#
# Load dataset và bỏ cột 'id' vì id là riêng biệt với mỗi người, không phải là feature

health_dataFrame = pd.read_csv("Data\\Dataset\\Nhom10_HealthcareStrokeDataset.csv")
health_dataFrame.drop(columns = "id", inplace = True)

# Xem thông tin về dataframe
# health_dataFrame.head()
health_dataFrame.info()

# Phân tích, thống kê data
# Một vài thông số
print(f'Tỉ lệ người mắc đột quỵ: {round(health_dataFrame["stroke"].value_counts(normalize = True)[1]*100, 3)}% ({health_dataFrame["stroke"].value_counts()[1]} người)')
print(f'Tỉ lệ người không mắc đột quỵ: {round(health_dataFrame["stroke"].value_counts(normalize = True)[0]*100, 3)}% ({health_dataFrame["stroke"].value_counts()[0]} người)')
print (f'\nTỉ lệ nữ: {round(health_dataFrame["gender"].value_counts(normalize = True)["Female"] * 100,3)} %')
print (f'Tỉ lệ nam: {round(health_dataFrame["gender"].value_counts(normalize = True)["Male"] * 100,3)} %')
print (f'Tỉ lệ giới tính khác: {round(health_dataFrame["gender"].value_counts(normalize = True)["Other"] * 100,3)} %')

# Phân tích data qua số liệu
def first_looking(col):
    print("----------------------------------------------------------------")
    print("Tên feature    : ", col)
    print("---------------------")
    print("Số lượng giá trị null   : ", health_dataFrame[col].isnull().sum())
    print("Số lượng kiểu phân loại : ", health_dataFrame[col].nunique())
    print(health_dataFrame[col].value_counts(dropna = False))

first_looking("gender")
first_looking("age")
first_looking("hypertension")
first_looking("heart_disease")
first_looking("ever_married")
first_looking("work_type")
first_looking("Residence_type")
first_looking("avg_glucose_level")
first_looking("bmi")
first_looking("smoking_status")
first_looking("stroke")



# Thống kê các giá trị thiếu - missing value
missing_number = health_dataFrame.isnull().sum().sort_values(ascending=False)
missing_percent = (health_dataFrame.isnull().sum() / health_dataFrame.isnull().count()).sort_values(ascending=False)
missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
print(missing_values)

health_dataFrame =  health_dataFrame.dropna(how='any',axis=0)

# Heat map: mối tương quan giữa các feature
sns.heatmap(health_dataFrame.corr(), annot=True);

# Phân tích qua biểu đồ thống kê và mối quan hệ từng feature với label (stroke)
# fig = px.histogram(health_dataFrame, x = "stroke", title = "Đột quỵ", width = 400, height = 400)
# fig.show()
fig = px.histogram(health_dataFrame, x = "gender", title = "Giới tính", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "gender", color = "stroke", title = "Giới tính - Đột Quỵ", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "age", title = "Tuổi", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "age", color = "stroke", title = "Tuổi - Đột Quỵ", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "hypertension", title = "Huyết áp", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "hypertension", color = "stroke", title = "Huyết áp - Đột quỵ", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "heart_disease", title = "Bệnh tim", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "heart_disease", color = "stroke", title = "Bệnh tim - Đột Quỵ", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "ever_married", title = "Tình trạng kết hôn", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "ever_married", color = "stroke", title = "Tình trạng kết hôn - Đột Quỵ", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "work_type", title = "Việc làm", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "work_type", color = "stroke", title = "Việc làm - Đột Quỵ", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "Residence_type", title = "Nơi ở", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "Residence_type", color = "stroke", title = "Nơi ở - Đột Quỵ", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "avg_glucose_level", title = "Đường trong máu", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "avg_glucose_level", color = "stroke", title = "Đường trong máu - Đột Quỵ", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "bmi", title = "Chỉ số khổi cơ thể (kg/m2)", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "bmi", color = "stroke", title = "Chỉ số khối cơ thể - Đột Quỵ", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "smoking_status", title = "Sử dụng thuốc lá", width = 400, height = 400)
fig.show()
fig = px.histogram(health_dataFrame, x = "smoking_status", color = "stroke", title = "Sử dụng thuốc lá - Đột Quỵ", width = 400, height = 400)
fig.show()
       

# Phân loại features
categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical = ['age','avg_glucose_level', 'bmi']

# one-hot-vector
one_hot_encoded_data = pd.get_dummies(health_dataFrame, columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
health_dataFrame = pd.concat([health_dataFrame,one_hot_encoded_data], axis = 1)
health_dataFrame.drop(columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"], axis=1, inplace=True)

health_dataFrame.info()

from sklearn.preprocessing import StandardScaler
x = health_dataFrame.drop("stroke",axis=1)
y = health_dataFrame["stroke"]
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size=0.2, random_state=42)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier()

randomForest.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
pred_rf = randomForest.predict(x_test)
accuracy = accuracy_score(pred_rf, y_test)
accuracy

# predict = randomForest.predict(x_scaled)

# counter = 0
# for i in predict:
#     if i[0] == 1:
#         counter = counter + 1
# print(counter)



# Hàm dự đoán dữ liệu thực sau khi đã training
# randomForest là cái có được sau khi train, dùng để dự đoán
def PredictStroke():
    predict_DataFrame = pd.read_csv(("Data\\Dataset\\Nhom10_Health_1.csv"))

    # Gán giá trị trống mặc định
    predict_DataFrame["gender"].fillna("Other", inplace=True)
    predict_DataFrame["age"].fillna(30, inplace=True)
    predict_DataFrame["hypertension"].fillna(0, inplace=True)
    predict_DataFrame["heart_disease"].fillna(0, inplace=True)
    predict_DataFrame["ever_married"].fillna(0, inplace=True)
    predict_DataFrame["work_type"].fillna("Private", inplace=True)
    predict_DataFrame["Residence_type"].fillna("Urban", inplace=True)
    predict_DataFrame["avg_glucose_level"].fillna(150.0, inplace=True)
    predict_DataFrame["bmi"].fillna(25, inplace=True)
    predict_DataFrame["smoking_status"].fillna("never smoked", inplace=True)

    one_hot_encoded_data = pd.get_dummies(predict_DataFrame, columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
    predict_DataFrame = pd.concat([predict_DataFrame, one_hot_encoded_data], axis = 1)
    predict_DataFrame.drop(columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"], axis=1, inplace=True)
    #predict_DataFrame = predict_DataFrame.iloc[5:]

    #predict_DataFrame.head()
    #randomForest.predict(predict_DataFrame)

    scaler_ = StandardScaler()
    scaler_.fit(predict_DataFrame)
    x_scaled_ = scaler.transform(predict_DataFrame)

    prediction = randomForest.predict(x_scaled_)
    
    counter = 0
    for i in prediction:
        if i[0] == 1:
            counter = counter + 1
    print(counter)


PredictStroke()