# 🛒 Ecommerce Sales Prediction

This project aims to predict the **Yearly Amount Spent** by customers on an ecommerce platform using various behavioral and demographic features. It's built as a part of an end-to-end machine learning pipeline, from data preprocessing to model deployment.

---

## 📂 Project Overview

In today's data-driven ecommerce world, understanding customer spending habits is essential. This project leverages regression techniques to predict how much a customer is likely to spend in a year, based on their interaction data.

---

## 📊 Dataset Description

The dataset contains the following columns:

| Feature Name             | Description                                                  |
|--------------------------|--------------------------------------------------------------|
| `Email`                  | Unique identifier for each user                              |
| `Address`                | User's address                                               |
| `Avatar`                 | User profile picture                                         |
| `Average Session Length` | Average time spent in a session on the app                   |
| `Time on App`            | Total time spent on the mobile app                           |
| `Time on Website`        | Total time spent on the website                              |
| `Length of Membership`   | Duration of user's membership with the platform              |
| `Yearly Amount Spent`    | **Target variable** — total yearly spending                  |

---

## 🧠 Objective

The primary goal is to build a regression model that accurately predicts the `Yearly Amount Spent` based on behavioral data.

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas & NumPy for data handling
- Matplotlib & Seaborn for data visualization
- Scikit-learn for model training
- Jupyter Notebooks for experimentation
- Git & GitHub for version control

---

## 🔎 Exploratory Data Analysis (EDA)

- Visualized relationships between features and the target
- Correlation heatmaps and scatter plots showed strong relationships with:
  - `Length of Membership`
  - `Time on App`
  - `Average Session Length`

---

## 🧪 Results

The best model was able to predict customer yearly spend with high accuracy, and key insights showed that **Length of Membership** and **Time on App** were strong predictors.

---
