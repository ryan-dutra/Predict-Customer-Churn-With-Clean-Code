# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This marks the inception of the inaugural project within Udacity's Machine Learning DevOps Engineer Nanodegree program. The primary goal of this endeavor is to develop refined code suitable for production while adhering to industry best practices. The focal point of the project revolves around predicting customer churn within the banking sector, constituting a classification task.

Outlined below is the proposed methodology:

* Data Loading and Exploration: Delve into a dataset comprising over 10,000 samples through exploratory data analysis (EDA).
* Data Preparation: Engage in feature engineering to craft 19 pertinent features conducive to training.
* Model Training: Employ two distinct classification models, namely sklearn's random forest and logistic regression.
* Feature Importance Analysis: Utilize the SHAP library to ascertain the most influential features impacting predictions, visually representing their effects.
* Model Persistence: Preserve the top-performing models along with their corresponding performance metrics.
* Code Refinement: Align the .py script file with PEP8 standards using the autopep8 module, further enhancing its clarity and readability. Additionally, ensure its cleanliness by attaining a score surpassing 8.0 via the pylint clean code module.

## Files and data description
The project is structured with the following directory architecture:

### Folders

- **Data**
  - **eda**: contains output of data exploration
  - **results**: contains the dataset in csv format
  - **images**: contains model scores, confusion matrix, ROC curve
  - **models**: contains saved models in .pkl format
  - **logs**: log generated during testing of churn_library.py file

### Project Files

- `churn_library.py`
- `churn_notebook.ipnyb`
- `requirements.txt`

### Test Files (unit test file and configuration files)

- `test_churn_script_logging_and_tests.py` 

## Running Files
- The project is designed to be executed using Python 3.8 along with the corresponding Python packages.
- All necessary packages are outlined in the `requirements.txt` file.
- To initiate the project, run the command `python churn_library.py` from the project directory.
- Alternatively, you can opt for a step-by-step approach by utilizing Jupyter Notebook.
- The effectiveness of the `churn_library.py` script was validated through testing conducted.


