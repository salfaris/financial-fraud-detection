# 💷 Payments Fraud Detection 🚨

#### Table of Contents

- [Project structure](#project-structure)
- [1. Experimentation](#1-experimentation)
  - [1.1. Fraud detection strategy research – paper reproduction.](#11-fraud-detection-strategy-research-paper-reproduction)
  - [1.2. From research to production model](#12-from-research-to-production-model)
- [2. Deployment](#2-deployment)
- [What was not covered](#what-was-not-covered)


## Project structure

This project consists of two key sections:

1. Experimentation
2. Deployment

## 1. Experimentation

### 1.1. Fraud detection strategy research – paper reproduction.

In the experimentation phase, my first focus is to reproduce and verify the class weight strategy by Aditya Oza in his preprint ["Fraud Detection using Machine Learning" (Oza, 2018)](https://www.semanticscholar.org/paper/Fraud-Detection-using-Machine-Learning-Oza-aditya/9f2c08d9efaa53cfabdd0ec47afa8015c7ff5bb9).

We can dive deeper into my findings for this reproduction, but the main ones include:

- The class weight strategy does work ✅ – one can attain a model that minimizes false positive rate such that recall is reasonably controllable.
- We did get precision and recall curve with shapes similar to that of the paper. ✅
- However, we did NOT get the exact same ideal class weights as the author. ❌ 
  - I suspect that this is related to multiple factors including unshared hyperparameters and random seeds. 
  - Furthermore, the ideal class weight obtained in the paper is done only via a single run (which can already take several hours; actually days for SVM with RBF kernel). I would argue we need to perform multiple runs and consider the uncertainty bands around this ideal class weight for experiment reporting.
- We obtained and and can draw the same conclusion from the precision-recall curves when comparing between models; with logistic regression being a superior lightweight choice and with SVM + RBF kernel only overperforming by a small margin. ✅

Considering the tradeoffs, we opted for the more lightweight logistic regression model to be used for deployment.

### 1.2. From research to production model

Oza's experiment did a granular analysis by training a model for each transaction type where there exists a fraudulent transaction. To recap, there are five transaction types:

1. `CASH_IN`: A cash deposit transaction via a merchant.
2. `CASH_OUT`: A cash withdrawal transaction via  merchant.
3. `DEBIT`: Similar to `CASH_OUT`. A cash withdrawal transaction from the mobile money service into a bank account.
4. `PAYMENT`: A payment transaction. Increases the balance of receiver and decreases the balance of the payee.
5. `TRANSFER`: Cash transfers between accounts using the mobile money platform.

In the dataset, fraudulent transactions only occurs through `CASH_OUT` and `TRANSFER` transaction types which means that two separate models (for each model type – logistic regression, linear svm and svm + rbf kernel) were trained.

Extending this to a production model, i.e., building models for each transaction type where we historically know we have fraudulent transaction means that we inherently introduce a rule-based system where if the transaction type is not `TRANSFER` or `CASH_OUT`, then we automatically label it as non-fraudulent; otherwise, we run the transaction rowset through the type-specific models.

Let's consider the pros and cons...

I would argue this is not scalable for both training and inference. If we have a new transaction type, e.g. `VOUCHER`, this means that we have to rerun the entire analysis and train a new model for the `VOUCHER` transaction type assuming fraudulent transactions occur in this type.

So we choose a double logistic regression for deployment...

## 2. Deployment

Now that we got a trained model, let's put this model into a working production environment.

The idea is that I will be simulating streaming payment transactions using the entire PaySim dataset and use the trained machine learning model to flag incoming transactions as fraud or not.

To visualize this process, I built a web app using Streamlit. The app is Dockerize with a Dockerfile and the image is pushed to the Google Artifact Registry. We then grabbed the pushed Docker image and deploy it using Google Cloud Platform (GCP) Cloud Run.

The deployed web app can be found here: https://fraud-web-app-qmjqqzknzq-ew.a.run.app/.


## What was not covered

Data onboarding was unfortunately not covered. I love data onboarding but unfortunately conjuring my own payments transaction dataset seems meaningless.