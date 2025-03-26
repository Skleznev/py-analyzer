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
n_models = 5  # Количество моделей

seeds = [42, 100, 2024, 999, 777]

# Загружаем модели перед инференсом
loaded_models = []
for seed in seeds:
    model_path = os.path.join(save_dir, f"catboost_model_{seed}_2.cbm")
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
    # Приводим к нижнему регистру
    word_lower = word.lower()
    return 1 if len(wordnet.synsets(word_lower)) > 0 else 0  # Возвращаем 1 или 0


with open('translit_cache.pkl', 'rb') as f:
    translit_words = pickle.load(f)


# Функция проверки юзернейма
def is_translit(username):
    return 1 if username.lower() in translit_words else 0  # Возвращаем 1 или 0


# Функция для отправки запроса к API
def get_valuations(domain):
    url = 'https://valuation.humbleworth.com/api/valuation'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json={'domains': [domain]}, headers=headers)
    if response.status_code == 200:
        return response.json().get('valuations', [])
    else:
        print(f"Ошибка: {response.status_code}")
        return []


url = 'https://api.coingecko.com/api/v3/simple/price'
params = {
    'ids': 'the-open-network',
    'vs_currencies': 'usd'
}
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()

course = data['the-open-network']['usd']


def get_data(username):
    start_time = time.time()

    # data = process_username(username)
    data = pd.DataFrame({'username': [username]})
    data['length'] = data['username'].apply(lambda x: len(x) if x is not None else 0)
    data['special_characters_length'] = data['username'].apply(evaluate_special_characters_length)
    data['numbers_length'] = data['username'].apply(evaluate_number_length)
    data['has_special_characters'] = data['username'].apply(has_special_characters)
    data['has_numbers'] = data['username'].apply(has_numbers)
    data['max_char_repeats'] = data['username'].apply(most_frequent_char_count)
    data['IsInDictionary_2'] = data['username'].apply(is_english_word_wordnet)
    data['is_translit'] = data['username'].apply(is_translit)

    # username = data['username'].iloc[0]
    domain = f"{username}.com"  # Формируем домен
    valuations = get_valuations(domain)
    if valuations:
        valuation_data = valuations[0]  # Первый (и единственный) элемент в списке
        data['auction'] = valuation_data.get('auction', None)
        data['brokerage'] = valuation_data.get('brokerage', None)
        data['marketplace'] = valuation_data.get('marketplace', None)
    else:
        # Если API не вернул данные, добавляем пустые столбцы
        data['auction'] = None
        data['brokerage'] = None
        data['marketplace'] = None

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Общее время выполнения: {total_time:.4f} сек")
    return data


def predict(X_test):
    predictions = np.array([model.predict(X_test) for model in loaded_models])
    uncertainty = np.std(predictions)
    pred_mean = max(1, np.mean(predictions))  # Защита от деления на 0
    confidence_score = (1 - uncertainty / pred_mean) * 100
    confidence_score = np.clip(confidence_score, 0, 100)

    result = "=" * 40

    print("=" * 40)
    result += "\n" + "📌 Предсказания моделей:"
    print("📌 Предсказания моделей:")
    for i, pred in enumerate(predictions, 1):
        pred_value = pred.item() if isinstance(pred, np.ndarray) else float(pred)  # Извлекаем число из массива
        result += "\n" + f"  Модель {i}: {pred_value * course:,.2f}"
        print(f"  Модель {i}: {pred_value * course:,.2f}")
    result += "\n" + "-" * 40
    print("-" * 40)
    result += "\n" + f"📊 Разброс предсказаний (uncertainty): {uncertainty * course:,.2f}"
    print(f"📊 Разброс предсказаний (uncertainty): {uncertainty * course:,.2f}")
    result += "\n" + f"📈 Среднее предсказание (pred_mean): {pred_mean * course:,.2f}"
    print(f"📈 Среднее предсказание (pred_mean): {pred_mean * course:,.2f}")

    result += "\n" +f"🔹 Уверенность модели: {confidence_score:.2f}%"
    print(f"🔹 Уверенность модели: {confidence_score:.2f}%")
    result += "\n" + "=" * 40
    print("=" * 40)
    return result


@functions_framework.http
def helloWorld(request: flask.Request) -> flask.typing.ResponseReturnValue:
    print(request.json)
    X_test = get_data(request.json["username"]).iloc[:, 1:]
    return predict(X_test)
