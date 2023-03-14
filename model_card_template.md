# Model Card

## Model Details
* Contributor: WU XIAOLI
* Model date: 2023-03-14
* Model version: 0.1
* Model type: Classification
* Training algorithm: Using Random Forest classifier to classify the income level of the people.
* Hyperparameters: n_estimators=100, max_depth=5, random_state=1
* Features: age, workclass, fnlgt, education, education_num, marital-status, occupation, 
relationship, race, sex, native-country, capital-gain, capital-loss, hours-per-week
* Target: income (>50K or <=50K)
* Contact: email to wuxiaol7@connect.hku.hk

## Intended Use
* The model is used to predict the income level of the people. The model can be used to help the government to make 
policies to help the people who have low income level to improve their income level. The model can also be used to help the people to know whether they can get a high income level or not.

## Training Data

## Evaluation Data

## Metrics
* Accuracy: 0.85
* Precision: 0.85
* Recall: 0.85
* F1 score: 0.85
## Ethical Considerations
* The model is used to predict the income level of the people. 
* The model can be used to help the government to make policies to 
help the people who have low income level to improve their income level.

## Caveats and Recommendations
* The model is trained on the data from 1994. The data may not be
representative of the current situation.
* The model is trained on the data from the United States. The model
may not be applicable to other countries.
