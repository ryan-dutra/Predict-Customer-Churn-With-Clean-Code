"""
    Useful functions for testing churn_notebook.py functions
    author: Ryan D. Abreu
    Date: May 4th 2024
"""

import os
import logging
from churn_library import *
# import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    Test data import
    '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    Test perform eda function
    '''
    data_frame = import_data("./data/bank_data.csv")
    try:
        perform_eda(data_frame)
        logging.info("Testing perform_eda: SUCCESS")
    except AttributeError as err:
        logging.error("Testing perform_eda: Input should be a dataframe")
        raise err
    except SyntaxError as err:
        logging.error("Testing perform_eda: Input should be a dataframe")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    Test encoder helper
    '''
    data_frame = import_data("./data/bank_data.csv")
    try:

        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        encoder_helper(dataframe=data_frame, category_lst=cat_columns)

        logging.info("Testing encoder_helper: SUCCESS")

    except KeyError as err:
        logging.error(
            "Testing encoder_helper: There are column names that doesn't exist in your dataframe")

    try:
        assert isinstance(cat_columns, list)
        assert len(cat_columns) > 0

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: category_lst argument should be a list with length > 0")
        raise err

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    Test perform_feature_engineering
    '''
    data_frame = import_data("./data/bank_data.csv")
    try:

        target = 'Churn'

        perform_feature_engineering(dataframe=data_frame,response=target)

        logging.info("Testing perform_feature_engineering: SUCCESS")

    except KeyError as err:
        logging.error(
            "Testing perform_feature_engineering: Target column names doesn't exist in dataframe")

    try:
        assert isinstance(target, str)

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: response argument should str")
        raise err


def test_train_models(train_models):
    '''
    Test train_models
    '''
    data_frame = import_data("./data/bank_data.csv")
    target = 'Churn'
    
    X_train, X_test, y_train, y_test = perform_feature_engineering(dataframe=data_frame, response=target)

    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except MemoryError as err:
        logging.error(
            "Testing train_models: Out of memory while train the models")
        raise err


if __name__ == "__main__":
    plt.ioff() 
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)