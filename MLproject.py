import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error as mae

lr = LinearRegression()

dd = pd.read_csv('C:/Users/jimin/Desktop/data/used-car.csv')

#print(dd.mean())
#print(dd.describe()) # 데이터셋 설명
#print(dd.info()) # 데이터셋 타입확인

for c in dd.columns: 
    if dd[c].dtype=='object': 
        dd[c] = dd[c].fillna('N')
        lbl = LabelEncoder()
        lbl.fit(list(dd[c].values))
        dd[c] = lbl.transform(dd[c].values)


d_input = dd[['model','year','transmission','mileage','fuelType','mpg','engineSize']].to_numpy()
d_target = dd['price'].to_numpy()

#print(lbl.classes_) #인코딩된 데이터 확인
#print(lbl.inverse_transform([0])) #0번은 어떤 데이터로 변환 되었는지 확인
     
print(dd.head()) # 변환 후 데이터 확인
print(dd.info()) # 데이터셋 타입확인

x_train, x_test, y_train, y_test = train_test_split(d_input,d_target,train_size=0.6)

#print(x_train.shape)
#print(x_test.shape)



lr.fit(x_train, y_train)

print(lr.coef_) # 기울기
print(lr.intercept_) # y절편

print("훈련세트 점수: {}%".format(round(lr.score(x_train, y_train)*100),1))
print("테스트세트 점수: {}%".format(round(lr.score(x_test, y_test)*100),1))

y_hat = lr.predict(x_test)
print("오차 값: {}".format(int(mae(y_hat,y_test)))) # 오차값

pred = int(lr.predict([[1,2016,0,69120,3,37.2,1.5]]))
c_pred = int(pred / 100 * 7)
print("입력하신 차량의 가격은 {}원으로 예상됩니다.".format(pred * 1300))

while True:
    type = input("차종을 입력하세요: ")
    age = int(input("나이를 입력하세요: "))
    if 20 <= age <= 23 and type == "승용차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(((pred * 1300) + (c_pred * 1300)) + 2236500)))
        break
    elif 20 <= age <= 23 and type == "경차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(pred * 1300 + 2236500)))
        break
    elif 24 <= age <= 25 and type == "승용차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(((pred * 1300) + (c_pred * 1300)) + 1898160)))
        break
    elif 24 <= age <= 25 and type == "경차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(pred * 1300 + 1898160)))
        break
    elif 26 <= age <= 29 and type == "승용차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(((pred * 1300) + (c_pred * 1300)) + 1473310)))
        break
    elif 26 <= age <= 29 and type == "경차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(pred * 1300 + 1473310)))
        break
    elif 30 <= age <= 34 and type == "승용차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(((pred * 1300) + (c_pred * 1300)) + 1404000)))
        break
    elif 30 <= age <= 34 and type == "경차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(pred * 1300 + 1404000)))
        break
    elif 35 <= age <= 42 and type == "승용차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(((pred * 1300) + (c_pred * 1300)) + 1391940)))
        break
    elif 35 <= age <= 42 and type == "경차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(pred * 1300 + 1391940)))
        break
    elif 43 <= age and type == "승용차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(((pred * 1300) + (c_pred * 1300)) + 1390000)))
        break
    elif 43 <= age and type == "경차":
        print("취등록세 보험료 포함 예상가격은 {}원 입니다.".format(np.round(pred * 1300 + 1390000)))
        break