import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.python.keras.utils.np_utils import to_categorical
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import load_model



df = pd.read_csv('./crawling_data/naver_headline_news_exam_total20241223.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.head())
print(df.info())
print(df.category.value_counts())

X = df['titles']
Y = df['category']

print(X[0])

# okt = Okt()
# okt_x = okt.morphs(X[0])

encoder = LabelEncoder()

#fit transform은 한번만 해야하고,  encoder는 저장을 해놓고, 그다음 추가 데이터가 구해지면 이 엔코더를 새로만드는것이 아니라 그대로 사용해야 한다
#엔코더 불러오기





#pickle은 텍스트 형태로 저장하는 것이 아니라 파일의 형태를 그대로 유지한다(바이너리 코드 형태로), 그래서 따로 읽어보는것은 불가하다
#wb = binary write
#with문은 with가 끝나면 닫는 명령어 (while문과 비슷)
with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)


label = encoder.classes_
print(label)

#fit.transform이 아니라 그냥 transform 왜냐하면 전에는 라벨을 만들어 준것이고,
# 이제는 이미 정해졌으니 그대로 라벨 변환 작업만 하는 것은 transform

labeled_y = encoder.transform(Y)
onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

okt = Okt()

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
print(X)


#불용어 제거
stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col = 0)
print(stopwords)

# 문장 수만큼 for문 돌리기
for sentence in range(len(X)):

    #사용할 단어 리스트 선언
    words = []

    #한문장의 형태소 수만큼 두번째 for문 돌리기
    for word in range(len(X[sentence])):
        #단어의 길이가 2이상일때만
        if len(X[sentence][word]) > 1:
            #stopwords에 있는 모든 단어 다 빼기, 1글자짜리 형태소 다빼기
            if X[sentence][word] not in list(stopwords['stopword']):
                words.append(X[sentence][word])

    # 다시 문장으로 조합하기
    X[sentence] = ' '.join(words)

print('X ====', X[:5])

token = Tokenizer()

#text를 기반으로한 tokenizer 리스트 만들기
#tokenizer도 이미 학습을 했으니 이전의 값들을 그대로 불러와서 사용해야 한다
#문제는 이전에 학습할때 사용하지 않은 형태소들은 '0' 처리해야한다

with open('./models/news_token_max_16.pickle', 'rb') as f:
    token = pickle.load(f)

#tokenizer 리스트를 문자구조로 만들기
tokened_X = token.texts_to_sequences(X)

#16개보다 형태소가 많은 경우 잘라서 버리기
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 16:
        #남는 뒷자리수 잘라서 버리기
        tokened_X[i] = tokened_X[i][:16]

x_pad = pad_sequences(tokened_X, 16)

#0을 사용해야하기 때문에 (빈 단어) 형태소 총 개수를 1을 더해줌
wordsize = len(token.word_index) + 1

print(wordsize)
print(tokened_X[0])

#문장마다 길이가 다르기 때문에 가장 긴 문장으로 맞추고 나머지 문장들의 빈단어 대신 0으로 채우되 0을 앞에다 채움

#최대 문장길이 찾는 알고리즘
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])

print('max words:', max)

#텐서플로우 함수 pad_sequences를 사용해서 0으로 채우기
X_pad = pad_sequences(tokened_X, max)

print(X_pad)


model = load_model('./models/news_category_classification_model_0.6317365169525146.h5')
# model = load_model('./models/news_category_classification_model_0.7395209670066833.h5')

preds = model.predict(X_pad)

predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    #argmax의 최대값을 지워서 두번째 값이 최고값이 되도록 수정
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predicts.append([most, second])

df['predict'] = predicts

print(df.head(30))

score = model.evaluate(X_pad, onehot_Y)
print(score[1])

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 1

print(df.OX.mean())
