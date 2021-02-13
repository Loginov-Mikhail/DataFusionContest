# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import io
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pickle

SLICE = 1000000

# Чтение датасета
df = pd.read_parquet('data/task1_test_for_user.parquet', columns=['id', 'item_name'])
df['item_name'] = df['item_name'].astype(str)
df['item_name'] = df['item_name'].str.lower()

# Чтение названий колонок результата работы модели
z = pd.read_csv('columns.csv')
dummiesColumnsName = list(z.columns)

# Загрузка словаря токенайзера
with open('tokenizer_item_name.json') as f:
    data = json.load(f)
    tokenizer_goods = tokenizer_from_json(data)

# Загрузка модели
model = pickle.load(open('model_v1.sav', 'rb'))

# Обработка названий товаров по полученному словарю
sequences_goods = tokenizer_goods.texts_to_sequences(df['item_name'])
max_position_len = 10
# Замена неожиданных значений None на 0
for i in range(len(sequences_goods)):
    sequences_goods[i] = [0 if v is None else v for v in sequences_goods[i]]

x1 = pad_sequences(sequences_goods, maxlen=max_position_len)

del sequences_goods

zFullPred = pd.DataFrame(columns=['pred'])
while len(x1) > SLICE:
    x = x1[:SLICE]
    y = model.predict(x)
    yPred = pd.DataFrame(y, columns=dummiesColumnsName)
    zPred = pd.DataFrame(pd.get_dummies(yPred).idxmax(1), columns=['pred'])
    zPred['pred'] = zPred['pred'].astype('uint8')
    zFullPred = zFullPred.append(zPred)
    x1 = x1[SLICE:]
x = x1
y = model.predict(x)
yPred = pd.DataFrame(y, columns=dummiesColumnsName)
zPred = pd.DataFrame(pd.get_dummies(yPred).idxmax(1), columns=['pred'])
zPred['pred'] = zPred['pred'].astype('uint8')
zFullPred = zFullPred.append(zPred)

zFullPred['pred'] = zFullPred['pred'].astype('uint8')
zFullPred['id'] = df['id']

zFullPred[['id', 'pred']].to_csv('answers.csv', index=False)
