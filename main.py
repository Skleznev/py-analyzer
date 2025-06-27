import numpy as np
import pandas as pd
import re
import ast
from typing_extensions import override
from collections import Counter
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import requests
from tqdm import tqdm
from transliterate import translit
from catboost import CatBoostRegressor
import time
import os
import pickle
import flask
import functions_framework


save_dir = 'models'
n_models = 5
seeds = [42, 100, 2024, 999, 777]


loaded_models = []
for seed in seeds:
    model_path = os.path.join(save_dir, f"catboost_model_log_{seed}_144.cbm")
    model = CatBoostRegressor()
    model.load_model(model_path)
    loaded_models.append(model)
    print(f"✅ Модель {seed} загружена из {model_path}")


def evaluate_special_characters_length(string):
    if string is None:
        return 0
    special_chars = re.findall(r"[^a-zA-Z0-9\s@]", string)
    return len(special_chars)


def evaluate_number_length(string):
    if string is None:
        return 0
    numbers = re.findall(r"\d", string)
    return len(numbers)


def has_special_characters(string):
    if string is None:
        return 0
    return 1 if re.search(r"[^a-zA-Z0-9\s@]", string) else 0


def has_numbers(string):
    if string is None:
        return 0
    return 1 if re.search(r"\d", string) else 0


def most_frequent_char_count(string):
    if string is None or string == '':
        return 0
    return Counter(string).most_common(1)[0][1]


def is_english_word_wordnet(word: str) -> int:
    word_lower = word.lower()
    return 1 if len(wordnet.synsets(word_lower)) > 0 else 0


with open('translit_cache.pkl', 'rb') as f:
    translit_words = pickle.load(f)


def is_translit(username):
    return 1 if username.lower() in translit_words else 0


def get_valuations(domain):
    url = 'https://valuation.humbleworth.com/api/valuation'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json={'domains': [domain]}, headers=headers)
    if response.status_code == 200:
        return response.json().get('valuations', [])
    else:
        print(f"Ошибка: {response.status_code}")
        return []


def get_data(username, ai_params):
    start_time = time.time()

    ai_columns = ['real_word','person','wordplay','company','mlt_letters','slang','betting','pop','sex','crypto','location','verb','offensive','name','a_b_c_d','web3','bot','Religion','Abbreviation']
    data = pd.DataFrame({'username': [username]})
    for col, value in zip(ai_columns, ai_params):
        data[col] = value

    data['length'] = data['username'].apply(lambda x: len(x) if x is not None else 0)
    data['special_characters_length'] = data['username'].apply(evaluate_special_characters_length)
    data['numbers_length'] = data['username'].apply(evaluate_number_length)
    data['has_special_characters'] = data['username'].apply(has_special_characters)
    data['has_numbers'] = data['username'].apply(has_numbers)
    data['max_char_repeats'] = data['username'].apply(most_frequent_char_count)
    data['IsInDictionary_2'] = data['username'].apply(is_english_word_wordnet)
    data['is_translit'] = data['username'].apply(is_translit)

    domain = f"{username}.com"
    valuations = get_valuations(domain)
    if valuations:
        valuation_data = valuations[0]
        data['auction'] = valuation_data.get('auction', None)
        data['brokerage'] = valuation_data.get('brokerage', None)
        data['marketplace'] = valuation_data.get('marketplace', None)
    else:
        data['auction'] = None
        data['brokerage'] = None
        data['marketplace'] = None

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Общее время выполнения: {total_time:.4f} сек")
    return data


def predict(X_test, username):
    predictions = np.array([np.expm1(model.predict(X_test)) for model in loaded_models])
    uncertainty = np.std(predictions)
    pred_mean = max(1, np.mean(predictions))  # Защита от деления на 0
    confidence_score = (1 - uncertainty / pred_mean) * 100
    confidence_score = np.clip(confidence_score, 0, 100)

    if pred_mean <= 100:
        score = (607 / 100) * pred_mean
    elif pred_mean <= 1000:
        score = 607 + ((845 - 607) / (1000 - 100)) * (pred_mean - 100)
    elif pred_mean <= 10000:
        score = 845 + ((977 - 845) / (10000 - 1000)) * (pred_mean - 1000)
    else:
        score = 977 + ((999 - 977) / (20000 - 10000)) * (pred_mean - 10000)
        score = min(score, 999)

    result = {
        "username": username,
        "priceInUSD": int(pred_mean),
        "confidence": int(round(confidence_score)),
        "score": int(round(score)),
        "version": "1.4.4"
    }

    return result

@functions_framework.http
def helloWorld(request: flask.Request) -> flask.typing.ResponseReturnValue:
    print(request.json)
    username = request.json["username"]
    ai_params = request.json["ai_params"]
    X_test = get_data(username, ai_params).iloc[:, 1:]
    prediction_result = predict(X_test, username)
    return flask.jsonify(prediction_result)
