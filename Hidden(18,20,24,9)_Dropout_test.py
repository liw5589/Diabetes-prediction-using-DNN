import tflearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

dataset = datasets.load_breast_cancer()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=12345)

enc = OneHotEncoder(handle_unknown='ignore')
y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()

n_inputs = 30
n_hidden1 = 3
n_hidden2 = 5
n_outputs = 2

n_epochs = 50
batch_size = 30

# 망
keeping_rate = 0.6

inputs = tflearn.input_data(shape=[None, n_inputs])
inputs_drop = tflearn.dropout(inputs, keeping_rate)
hidden1 = tflearn.fully_connected(inputs_drop, n_hidden1, activation='relu', name='hidden1')
hidden1_drop = tflearn.dropout(hidden1, keeping_rate)

hidden2 = tflearn.fully_connected(hidden1_drop, n_hidden2, activation='relu',name='hidden2')
hidden2_drop = tflearn.dropout(hidden2 , keeping_rate)

softmax = tflearn.fully_connected(hidden2_drop, n_outputs, activation='softmax', name ='output')
net = tflearn.regression(softmax)

# 모델 객체 생성
model = tflearn.DNN(net)

model.load("드롭아웃 시킨거 불로오기")
acc_test = model.evaluate(x_test,y_test,batch_size)
print("테스트 데이터 : " + str(acc_test))