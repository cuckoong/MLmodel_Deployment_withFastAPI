# Model Card

## Model Details
* Contributor: WU XIAOLI
* Model date: 2023-03-14
* Model version: 0.1
* Model type: Classification
* Training algorithm: Using Random Forest classifier to classify the income level of the people.
* Hyperparameters: max_depth=15, min_samples_leaf=1, min_samples_split=5, n_estimators=300
* Features: age, workclass, fnlgt, education, education_num, marital-status, occupation, 
relationship, race, sex, native-country, capital-gain, capital-loss, hours-per-week
* Target: income (>50K or <=50K)
* Contact: email to wuxiaol7@connect.hku.hk

## Intended Use
* The model is used to predict the income level of the people. The model can be used to help the government to make 
policies to help the people who have low income level to improve their income level. The model can also be used to help the people to know whether they can get a high income level or not.

## Training Data
* source: census.csv provided by the course instructor
* size: 32561 rows and 15 columns
* Training size: 80% of the data
* Training Target: income (>50K or <=50K)
* Training Inputs: age, workclass, fnlgt, education, education-num,
       marital-status, occupation, relationship, race, sex,
       capital-gain, capital-loss, hours-per-week, native-country.

## Evaluation Data
* source: census.csv provided by the course instructor
* Testing size: 20% of the data
* Testing Target: income (>50K or <=50K)

## Metrics
* Accuracy: 0.672
* Precision: 0.793
* Recall: 0.585
* F1 score: 0.673

## Ethical Considerations
* The model is used to predict the income level of the people. 
* The model can be used to help the government to make policies to 
help the people who have low income level to improve their income level.

## Caveats and Recommendations
* The model is trained on the data from the United States. The model
may not be applicable to other countries.
