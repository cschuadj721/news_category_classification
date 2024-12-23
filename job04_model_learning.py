import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, LSTM, Embedding, Conv1D, MaxPool1D
# from keras.layers import *
# from keras.models import *
from tensorflow.python.framework.test_ops import kernel_label

X_train = np.load('./crawling_data/news_data_X_train__wordsize6348.npy', allow_pickle = True)
X_test = np.load('./crawling_data/news_data_X_test__wordsize6348.npy', allow_pickle = True)
Y_train = np.load('./crawling_data/news_data_Y_train__wordsize6348.npy', allow_pickle = True)
Y_test = np.load('./crawling_data/news_data_Y_test__wordsize6348.npy', allow_pickle = True)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()

# 자연어 학습일 때의 시작 레이어 전체 입력데이터의 모든 형태소의 개수 차원의 의미공간에 각각형태소를 벡터화해주는 레이어
model.add(Embedding(6348, 300, input_length = 16))

model.build(input_shape = (None,16))

model.add(Conv1D(32, kernel_size=5, padding = 'same', activation = 'relu'))
model.add(MaxPool1D(pool_size=1))

#return sequences true 를 주면  RNN에서 나오는 각 계산값을 모아서 전달한다는 말이고,
#모델이 완성된 경우에는 각계산값들은 더이상 필요가 없고 최종 RNN 학습을 통해서 생성된 하나의 파라미터 값만이 사용된다
model.add(LSTM(128, activation = 'tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation = 'tanh', return_sequences=True))
model.add(Dropout(0.3))

# 마지막 RNN, LSTM 층에서는 return sequences true가 필요없다 (전달할 필요가 없으니)
# 그러나 전달해주어야 할 필요가 있을때도 있다
model.add(LSTM(64, activation = 'tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size = 128,
                     epochs = 10, validation_data = (X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose = 0)
print('Final test set accuracy: ', score[1])

model.save('./models/news_category_classification_model_{}.h5'.format(
    fit_hist.history['val_accuracy'][-1]))

plt.plot(fit_hist.history['val_accuracy'], label = 'val_accuracy')
plt.plot(fit_hist.history['accuracy'], label = 'accuracy')
plt.legend()
plt.show()
