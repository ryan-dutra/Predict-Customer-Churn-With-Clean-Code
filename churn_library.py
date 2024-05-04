"""
    Useful functions for churn_notebook.py
    author: Ryan D. Abreu
    Date: May 4th 2024
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    Returns a pandas dataframe for the CSV file located at the specified path.

    Parameters:
        pth(str): The path to the CSV file
    Returns:
        dataframe:  Pandas dataframe containing the data 
    '''
    dataframe = pd.read_csv(pth, index_col=0)

    # Encode the churn-dependent variable: 0 for "No churn"; 1 for "Churned"
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Drop redudant variables
    dataframe.drop('Attrition_Flag', axis=1, inplace=True)

    # Drop unusefull variables
    dataframe.drop('CLIENTNUM', axis=1, inplace=True)

    return dataframe


def perform_eda(dataframe, display=False):
    '''
    Conducts EDA on a df and saves the resulting figures to the images folder.

    Parameters:
        dataframe: Pandas dataframe containing the data to be analyzed
        display: Boolean indicating whether to display the plots (default is False)

    Returns:
        None
    '''

    # Analyze categorical attributes and visualize their distribution
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()
    for cat_column in cat_columns:
        plt.figure(figsize=(7, 4))
        (dataframe[cat_column]
            .value_counts('normalize')
            .plot(kind='bar',
                  rot=45,
                  title=f'{cat_column} - % Churn')

         )
        plt.savefig(os.path.join("./images/eda", f'{cat_column}.png'),
                    box_inches='tight')
        if display:
            plt.show()
        plt.close()

    # Dealing with Numeric features
    plt.figure(figsize=(10, 5))
    (dataframe['Customer_Age']
        .plot(kind='hist',
              title='Distribution of Customer Age')
     )
    plt.savefig(os.path.join("./images/eda", 'Customer_Age.png'),
                box_inches='tight')
    if display:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    # Show distribution
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    if display:
        plt.show()
    plt.close()

    # plot correlation matrix
    plt.figure(figsize=(15, 7))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join("./images/eda", 'correlation_matrix.png'),
                box_inches='tight')
    if display:
        plt.show()
    plt.close()
 
    plt.figure(figsize=(15, 7))
    (dataframe[['Total_Trans_Amt', 'Total_Trans_Ct']]
        .plot(x='Total_Trans_Amt',
              y='Total_Trans_Ct',
              kind='scatter',
              title='Correlation analysis between 2 features')
     )
    if display:
        plt.show()
    plt.close()


def encoder_helper(dataframe, category_lst, response='Churn'):
    '''
    Helper function that transforms each categorical column into a new column representing the proportion of churn for each category.
    Parameters:
        dataframe: Pandas dataframe containing the data
        category_lst: List of columns that contain categorical features
        response:  String indicating the response name (optional). 

    Returns:
        dataframe: Pandas dataframe with new columns representing the proportion of churn for each category
    '''
    for category in category_lst:
        category_groups = dataframe.groupby(category).mean()[response]
        new_feature = category + '_' + response
        dataframe[new_feature] = dataframe[category].apply(
            lambda x: category_groups.loc[x])

    # Drop the obsolete categorical features of the category_lst
    dataframe.drop(category_lst, axis=1, inplace=True)

    return dataframe


def perform_feature_engineering(dataframe, response='Churn'):
    '''
    Converts remaining categorical using one-hot encoding adding the response
    str prefix to new columns Then generate train and test datasets

    Parameters:
        dataframe: Pandas dataframe
        response: String of response name [optional argument that
        could be used for naming variables or index y column]

    Returns:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''

    # Get categorical features
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()

    # Encode categorical features using mean of response variable on category
    dataframe = encoder_helper(dataframe, cat_columns, response='Churn')

    y = dataframe[response]
    X = dataframe.drop(response, axis=1)
    # Spliting data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def plot_classification_report(model_name,
                               y_train,
                               y_test,
                               y_train_preds,
                               y_test_preds,
                               display=False):
    '''
    Produces classification report for training and testing results and stores
    report as image in images folder

    Parameters:
        model_name(str): Name of the model, ie 'Random Forest'
        y_train: Training response values
        y_test:  Test response values
        y_train_preds: Training predictions from model_name
        y_test_preds: Test predictions from model_name
        display: Boolean indicating whether to display the plots (default is False)

    Returns:
        None
    '''

    plt.rc('figure', figsize=(5, 5))

    # Classification report for Training dataset
    plt.text(0.01, 1.25,
             str(f'{model_name} Train'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    # Classification report for Tresting dataset
    plt.text(0.01, 0.6,
             str(f'{model_name} Test'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    plt.axis('off')

    # Save figure
    fig_name = f'Classification_report_{model_name}.png'
    plt.savefig(
        os.path.join(
            "./images/results",
            fig_name),
        bbox_inches='tight')

    # Display figure
    if display:
        plt.show()
    plt.close()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produces classification report for training and testing results and stores
    report as image in images folder using plot_classification_report
    helper function

    Parameters:
        y_train: Training response values
        y_test:  Test response values
        y_train_preds_lr: Training predictions from logistic regression
        y_train_preds_rf: Training predictions from random forest
        y_test_preds_lr: Test predictions from logistic regression
        y_test_preds_rf: Test predictions from random forest

    Returns:
                     None
    '''

    plot_classification_report('Logistic Regression',
                               y_train,
                               y_test,
                               y_train_preds_lr,
                               y_test_preds_lr)
    plt.close()

    plot_classification_report('Random Forest',
                               y_train,
                               y_test,
                               y_train_preds_rf,
                               y_test_preds_rf)
    plt.close()


def feature_importance_plot(model, X_data, model_name, output_pth, display=False):
    '''
    Creates and stores the feature importances in pth

    Parameters:
        model: Model object containing feature_importances_
        X_data: Pandas dataframe of X values
        output_pth(str): Path to store the figure
        display: Boolean indicating whether to display the plots (default is False)

    Returns:
                     None
    '''

    # Get feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title(f"Feature Importance for {model_name}")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    fig_name = f'feature_importance_{model_name}.png'
    plt.savefig(os.path.join(output_pth, fig_name), bbox_inches='tight')

    # display feature importance figure
    if display:
        plt.show()
    plt.close()


def confusion_matrix(model, model_name, X_test, y_test, display=False):
    '''
        Display confusion matrix of a model on test data

        Parameters:
            model: Trained model
            X_test: X testing data
            y_test: y testing data
            display: Boolean indicating whether to display the plots (default is False)
        Returns:
            None
        '''
    class_names = ['Not Churned', 'Churned']
    plt.figure(figsize=(15, 5))
    ax = plt.gca()
    plot_confusion_matrix(model,
                          X_test,
                          y_test,
                          display_labels=class_names,
                          cmap=plt.cm.Blues,
                          xticks_rotation='horizontal',
                          ax=ax)
    # Hide grid lines
    ax.grid(False)
    plt.title(f'{model_name} Confusion Matrix on test data')
    plt.savefig(
        os.path.join(
            "./images/results",
            f'{model_name}_Confusion_Matrix'),
        bbox_inches='tight')
    if display:
        plt.show()
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    Train and store model results

    Parameters:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    Returns:
        None
    '''
    # Initialize the Random Forest algorithm
    rfc = RandomForestClassifier(random_state=42)
    # Initialize the Logistic Regression algorithm
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    # GS for random forest parameters and instantiation
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    # Training Random Forest using GS
    cv_rfc.fit(X_train, y_train)
    # Training Logistic Regression
    lrc.fit(X_train, y_train)
    # get predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # calculate classification scores
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # plot ROC-curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8
    )
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    # Saving ROC-curves to images directory
    plt.savefig(
        os.path.join(
            "./images/results",
            'ROC_curves.png'),
        bbox_inches='tight')
    plt.close()
    # Saving the best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    for model, model_type in zip([cv_rfc.best_estimator_, lrc],
                                 ['Random_Forest', 'Logistic_Regression']
                                 ):
        # Display confusion matrix on test data
        confusion_matrix(model, model_type, X_test, y_test)

    # Display feature importance on train data
    feature_importance_plot(cv_rfc.best_estimator_,
                            X_train,
                            'Random_Forest',
                            "./images/results")


if __name__ == "__main__":
    PATH = "./data/bank_data.csv"
    DATASET = import_data(PATH)
    perform_eda(DATASET)
    X_train, X_test, y_train, y_test = perform_feature_engineering(DATASET, response='Churn')
    train_models(X_train, X_test, y_train, y_test)
   