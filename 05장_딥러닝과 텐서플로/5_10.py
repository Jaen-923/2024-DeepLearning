# 손실 함수의 성능 비교 실험
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# MNIST 읽어와서 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32)/255.0 # 0~255의 범위를 0~1 범위로 정규화
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10) # 원핫코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 신경망 구조 설정
n_input = 784
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 512
n_hidden4 = 512
n_output = 10

# MSE
dmlp_mse = Sequential()
dmlp_mse.add(Dense(units=n_hidden1, activation='tanh', input_shape=(n_input,)))
dmlp_mse.add(Dense(units=n_hidden2, activation='tanh'))
dmlp_mse.add(Dense(units=n_hidden3, activation='tanh'))
dmlp_mse.add(Dense(units=n_hidden4, activation='tanh'))
dmlp_mse.add(Dense(units=n_output, activation='softmax'))
dmlp_mse.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
hist_mse = dmlp_mse.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=2)

# cross entropy
dmlp_ce = Sequential()
dmlp_ce.add(Dense(units=n_hidden1, activation='tanh', input_shape=(n_input,)))
dmlp_ce.add(Dense(units=n_hidden2, activation='tanh'))
dmlp_ce.add(Dense(units=n_hidden3, activation='tanh'))
dmlp_ce.add(Dense(units=n_hidden4, activation='tanh'))
dmlp_ce.add(Dense(units=n_output, activation='softmax'))
dmlp_ce.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
hist_ce = dmlp_ce.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=2)

# 두 모델의 정확률 비교
res_mse = dmlp_mse.evaluate(x_test, y_test, verbose=0)
print("평균제곱오차의 정확률은", res_mse[1]*100)
res_ce = dmlp_ce.evaluate(x_test, y_test, verbose=0)
print("교차 엔트로피의 정확률은", res_ce[1]*100)

# 하나의 그래프에서 두 모델을 비교
plt.plot(hist_mse.history['accuracy'])
plt.plot(hist_mse.history['val_accuracy'])
plt.plot(hist_ce.history['accuracy'])
plt.plot(hist_ce.history['val_accuracy'])
plt.title('Model accuarcy comparison between MSE and cross entropy')
plt.ylabel('Accuarcy')
plt.xlabel('Epoch')
plt.legend(['Train_mse', 'Validation_mse', 'Train_ce', 'Validation_ce'], loc='best')
plt.grid()
plt.show()
