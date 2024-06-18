## Overview
This repository presents the parameters of the models used in the study titled "Comparative Analysis of Deep Neural Networks and Machine Learning for Detecting Malicious Domain Name Registrations". The study evaluates various models to enhance the detection of malicious domain name registrations. 

## Textual Models

| Model        | Training Parameters                                                                                        |
|--------------|-----------------------------------------------------------------------------------------------------------|
| Canine       | `batch_size = 128, BASE_LR = 2e-5, OPT = 'adam'`                                                            |
| BERT         | `batch_size = 128, BASE_LR = 2e-5, OPT = 'adam'`                                                            |
| RoBERTa      | `batch_size = 128, BASE_LR = 2e-5, OPT = 'adam'`                                                            |
| ALBERT       | `batch_size = 128, BASE_LR = 2e-5, OPT = 'adam'`                                                            |
| MobileBERT   | `batch_size = 128, BASE_LR = 2e-5, OPT = 'adam'`                                                            |
| MLP (TFIDF)  | `batch_size = 500, BASE_LR = 0.005, OPT = 'adam'`                                                           |
| LR (TFIDF)   | `C=10`                                                                                                    |

## Numeric Models

| Model    | Training Parameters                                                                                        |
|----------|-----------------------------------------------------------------------------------------------------------|
| RF       | `n_estimators=50, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=10, bootstrap=True` |
| SVM      | `C=10`                                                                                                    |
| KNN      | `weights='uniform', n_neighbors=9, metric='minkowski'`                                                    |
| XGBOOST  | `tree_method='hist', subsample=0.7, n_estimators=300, max_depth=6, learning_rate=0.01, colsample_bytree=0.7, device="cuda"` |
| LR       | `C=100`                                                                                                   |

## Both Textual and Numeric Models

| Model         | Training Parameters                                                                                        |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Canine        | `batch_size = 160, BASE_LR = 2e-5, OPT = 'adam'`                                                            |
| BERT          | `batch_size = 160, BASE_LR = 2e-5, OPT = 'adam'`                                                            |
| RoBERTa       | `batch_size = 160, BASE_LR = 2e-5, OPT = 'adam'`                                                            |
| ALBERT        | `batch_size = 160, BASE_LR = 2e-5, OPT = 'adam'`                                                            |
| MobileBERT    | `batch_size = 160, BASE_LR = 2e-5, OPT = 'adam'`                                                            |
| MLP + (TFIDF) | `batch_size = 500, BASE_LR = 0.005, OPT = 'adam'`                                                           |
| RF + (TFIDF)  | `max_depth=400`                                                                                           |
| SVM + (TFIDF) | `gamma=0.01, C=100`                                                                                       |
| KNN + (TFIDF) | `weights='uniform', n_neighbors=10, metric='minkowski'`                                                    |
| XGBOOST + (TFIDF) | `tree_method='hist', n_estimators=300, max_depth=6, learning_rate=0.3`                                     |
| LR + (TFIDF)  | `C=20`                                                                                                    |
