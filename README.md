# Fake News Detection Web App

A machine learning-based web application built using Flask that classifies news headlines as **Real** or **Fake** using Natural Language Processing.

## Project Overview

This project uses a trained machine learning model to analyze news headlines and predict whether they are real or fake. Users can enter a news headline in the web app and get an instant prediction.

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML/CSS

## Features

- Real-time news headline classification
- Simple and clean user interface
- Machine learning model integration
- NLP-based text processing using TF-IDF

## How It Works

1. User enters a news headline.
2. The text is converted into numerical features using TF-IDF Vectorizer.
3. The trained Logistic Regression model predicts whether the headline is real or fake.
4. The result is displayed on the web page.

## Dataset

This project uses the `FakeNewsNet.csv` dataset.

The model is currently trained on the `title` column, so it works as a news headline classifier.

## Model Performance

After training the model, accuracy is displayed in the terminal.

Example:

```txt
Accuracy: 0.__
