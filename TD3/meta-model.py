import numpy as np

import requests

import time

from sklearn import datasets

import pandas as pd

from sklearn.model_selection import train_test_split

import json

import os

 

# Config

INITIAL_STAKE = 1000 # Initial euros for each validator

CHECK_INTERVAL = 2   # Seconds between checks

REWARD = 10          # Reward for correct prediction

SLASH = 10           # Penalty for wrong prediction

 

iris_classes = ["setosa","versicolor","virginica"]

 

 

# JSON file to track validator balances

DATABASE_FILE = "validator_balances.json"

 

class PosValidator:

    def __init__(self, url, initial_stake):

        self.url = url

        self.stake = initial_stake

        self.predictions = []

 

def run_pos_network(validator_urls, test_data):

    # Init validators

    validators = [PosValidator(url, INITIAL_STAKE) for url in validator_urls]

   

    while True:

        print("Current stakes:")

        for i,v in enumerate(validators):

            print(f"Validator {i}: {v.stake:.2f} euros")

       

        # Randomly select a test sample

        idx = np.random.randint(len(test_data))

        sample = test_data.iloc[idx]

        sample = {

            "sepal_length":sample[0],

            "sepal_width":sample[1],

            "petal_length":sample[2],

            "petal_width":sample[3]

        }

       

        # Get the predictions for each validator

        for i,v in enumerate(validators):

            response = requests.get(v.url, params=sample)

            response_json = response.json()

            v.predictions.append(response_json["probability_scores"])

       

        # Get weighted mean of predictions

        mean_predictions = [0,0,0]

        total_stake = sum(v.stake for v in validators)

        for j in range(3):

            for i,v in enumerate(validators):

                mean_predictions[j] += v.predictions[-1][j] * (v.stake / total_stake)

       

        # Get most voted class and reward/punish validators

        final_pred = np.argmax(mean_predictions)

        print(f"Prediction: {iris_classes[final_pred]}")

        print()

        for i,v in enumerate(validators):

            if v.predictions[-1][final_pred] == max(v.predictions[-1]):

                v.stake += REWARD

            else:

                v.stake -= SLASH

       

        # Update balances in JSON database

        balances = {v.url: v.stake for v in validators}

        save_balances(balances)

       

        # Wait a bit

        time.sleep(CHECK_INTERVAL)

 

def main():

    # Initialize our validators

    validator_urls = [

        "https://bbc0-89-30-29-68.ngrok-free.app/predict",

        "https://1b00-89-30-29-68.ngrok-free.app/predict",

        "https://0a6a-89-30-29-68.ngrok-free.app/predict"

    ]

   

    iris = datasets.load_iris()

    iris = pd.DataFrame(

        data= np.c_[iris['data'], iris['target']],

        columns= iris['feature_names'] + ['target']

    )

    X = iris.drop('target', axis=1)

    y = iris['target']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

   

    run_pos_network(validator_urls, X_test)

 

# Function to load validator balances

def load_balances(validator_urls):

    if os.path.exists(DATABASE_FILE):

        with open(DATABASE_FILE, "r") as file:

            balances = json.load(file)

    else:

        # Initialize balances if file doesn't exist

        balances = {url: INITIAL_STAKE for url in validator_urls}

        save_balances(balances)

    return balances

 

# Function to save validator balances

def save_balances(balances):

    with open(DATABASE_FILE, "w") as file:

        json.dump(balances, file, indent=4)