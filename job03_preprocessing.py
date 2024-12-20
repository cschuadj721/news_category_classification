import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.python.keras.utils.np_utils import to_categorical
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



df = pd.read_csv('./crawling_data/naver_headline_news_exam_total20241220.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.head())
print(df.info())
print(df.category.value_counts())

X = df['titles']
Y = df['category']

print(X[0])

okt = Okt()
okt_x = okt.morphs(X[0])

encoder = LabelEncoder()

#fit transform은 한번만 해야하고,  encoder는 저장을 해놓고, 그다음 추가 데이터가 구해지면 이 엔코더를 새로만드는것이 아니라 그대로 사용해야 한다
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:3])

label = encoder.classes_
print(label)

#pickle은 텍스트 형태로 저장하는 것이 아니라 파일의 형태를 그대로 유지한다(바이너리 코드 형태로), 그래서 따로 읽어보는것은 불가하다
#wb = binary write
#with문은 with가 끝나면 닫는 명령어 (while문과 비슷)
with open('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

#토큰화 -> 형태소 단위로 모든 단어들을 나누는것
#원형화 -> 모든 형태소를 원형으로 바꾸는 것

onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

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
token.fit_on_texts(X)

#tokenizer 리스트를 문자구조로 만들기
tokened_X = token.texts_to_sequences(X)

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

X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size = 0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

np.save('./crawling_data/news_data_X_train_max_{}_wordsize{}'.format(max, wordsize), X_train)
np.save('./crawling_data/news_data_Y_train_max_{}_wordsize{}'.format(max, wordsize), Y_train)
np.save('./crawling_data/news_data_X_test_max_{}_wordsize{}'.format(max, wordsize), X_test)
np.save('./crawling_data/news_data_Y_test_max_{}_wordsize{}'.format(max, wordsize), Y_test)