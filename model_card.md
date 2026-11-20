# Model Card

## Model Details
This model is a supervised binary classification model built using scikit-learn's `GradientBoostingClassifier` (with `random_state=42` and default hyperparameters) to predict whether an individual's annual income exceeds $50,000. It was developed in November 2026 as part of the Udacity Machine Learning DevOps Engineer Nanodegree program. Categorical input features are transformed using `OneHotEncoder(sparse_output=False, handle_unknown="ignore")`, continuous features are passed directly without scaling, and the target label is encoded using `LabelBinarizer`.

## Intended Use
This model is intended for educational and portfolio demonstration purposes to showcase end-to-end machine learning pipeline construction, slice-based fairness evaluation, continuous integration, and RESTful API deployment with FastAPI. The primary intended users are machine learning practitioners, students, and reviewers evaluating MLOps workflows. Out-of-scope use cases include making real-world financial, lending, hiring, insurance, or credit decisions about individuals.

## Training Data
The model was trained on 80% of the UCI Census Income dataset (`census.csv`, 32,561 total records), resulting in 26,048 training rows created via a random split (`test_size=0.20`, `random_state=42`). The dataset originates from the 1994 U.S. Census database and contains 14 demographic and employment features (including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country) along with a binary salary target (`<=50K` or `>50K`).

## Evaluation Data
The model was evaluated on the remaining 20% holdout test split (6,513 records) partitioned from `census.csv` prior to fitting any encoders or estimators. During evaluation, the test set was transformed using the fitted `OneHotEncoder` and `LabelBinarizer` from the training split (`training=False`) to prevent data leakage.

## Metrics
The model is evaluated using Precision, Recall, and the F1 score (`fbeta_score` with `beta=1`), which are well suited for imbalanced binary classification where false positives and false negatives carry distinct trade-offs. On the holdout test dataset, the model achieved a Precision of 0.8030, a Recall of 0.6225, and an F1 score of 0.7013. Disaggregated slice metrics across all unique values of every categorical feature are recorded in `slice_output.txt`.

## Ethical Considerations
The Census Income dataset contains sensitive demographic attributes including race, sex, marital status, and native country, as well as historical socioeconomic disparities from 1994 U.S. Census records. Because class distributions and income outcomes vary significantly across demographic slices, deploying this model in real-world decision-making systems could perpetuate or amplify historical biases against underrepresented groups. Slice-based evaluation was performed to surface disparities across categorical subgroups before considering any production use.

## Caveats and Recommendations
Because the underlying dataset was collected in 1994, its income thresholds ($50,000) and labor market patterns do not reflect current economic conditions or modern demographic distributions. Additionally, some categorical slices (such as rare workclasses or native countries) contain very few evaluation samples, which can cause high variance in slice-level precision and recall scores. For any practical application beyond educational demonstration, the model should be retrained on current, representative data, evaluated with confidence intervals across intersectional subgroups, and audited for fairness constraints.
