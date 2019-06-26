
#20150145 김민석
#건강검진기록 데이터로 당뇨병을 예측하기

import tflearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#pandas 라이브러리로 csv 가져옴
Health = pd.read_csv('HealthInformation/NHIS_OPEN_GJ_2016.csv', encoding = 'euc-kr')

#필요없는거 싹다 날림
Health = Health.drop('기준년도',1)
Health = Health.drop('가입자일련번호',1)
Health = Health.drop('데이터공개일자',1)
Health = Health.drop('시도코드',1)
Health = Health.drop('청력(좌)',1)
Health = Health.drop('청력(우)',1)
Health = Health.drop('수축기혈압',1)
Health = Health.drop('이완기혈압',1)
Health = Health.drop('총콜레스테롤',1)
Health = Health.drop('트리글리세라이드',1)
Health = Health.drop('HDL콜레스테롤',1)
Health = Health.drop('LDL콜레스테롤',1)
Health = Health.drop('혈색소',1)
Health = Health.drop('혈청크레아티닌',1)
Health = Health.drop('(혈청지오티)AST',1)
Health = Health.drop('(혈청지오티)ALT',1)
Health = Health.drop('감마지티피',1)
Health = Health.drop('음주여부',1)
Health = Health.drop('구강검진 수검여부',1)
Health = Health.drop('치아우식증유무',1)
Health = Health.drop('결손치유무',1)
Health = Health.drop('치아마모증유무',1)
Health = Health.drop('제3대구치(사랑니)이상',1)
Health = Health.drop('치석',1)
Health = Health.drop('요단백',1)
Health = Health.drop('시력(좌)',1)
Health = Health.drop('시력(우)',1)

#숫자가 아닌 값은 0으로 바꿈
Health = Health.fillna(0).astype(int)

# 당뇨병의 기준 -> 공복혈당이 126 이상
# 공복혈당이 130이 넘는다면 당뇨병 (1) 넘지 않는다면 당뇨병이 아님 (0) 으로 데이터를 바꿈
# 이제 식전혈당(공복혈당) 당뇨병의 유무에 의한 값으로 바뀜
Health['식전혈당(공복혈당)'] = np.where(Health['식전혈당(공복혈당)'] > 120, 1, 0)

# 당뇨병 유무의 칼럼을 label에 저장하고 csv파일에서 당뇨병 유무의 칼럼을 뺀 나머지는 data에 저장

data = Health.drop('식전혈당(공복혈당)',1)
label = Health['식전혈당(공복혈당)']

x_data = data.as_matrix()
y_data = label.as_matrix()


#학습데이터 테스트 데이터 분리 하기
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size= 0.8, random_state=30)

enc = OneHotEncoder(handle_unknown='ignore')
y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()

n_inputs = 6   #[성별코드, 연령대코드, 신장, 체중, 허리둘레, 흡연상태]
n_hidden1 = 18
n_hidden2 = 20
n_hidden3 = 24
n_hidden4 = 9
n_outputs = 2     #마지막에 결과는 당뇨병 이다 / 아니다 이기때문에 -> 2개

n_epochs = 50
batch_size = 128

# 망

inputs = tflearn.input_data(shape=[None, n_inputs])
hidden1 = tflearn.fully_connected(inputs, n_hidden1, activation='relu', name='hidden1')
hidden2 = tflearn.fully_connected(hidden1, n_hidden2, activation='relu',name='hidden2')
hidden3 = tflearn.fully_connected(hidden2, n_hidden3,activation='relu',name = 'hidden3')
hidden4 = tflearn.fully_connected(hidden3, n_hidden4,activation='relu',name = 'hidden4')
softmax = tflearn.fully_connected(hidden4, n_outputs, activation='softmax', name ='output')
net = tflearn.regression(softmax)

# 모델 객체 생성
model = tflearn.DNN(net)

model.load("train_model/hidden(18,20,24,9).tfl")
acc_test = model.evaluate(x_test,y_test,batch_size)
print("테스트 데이터 : " + str(acc_test))