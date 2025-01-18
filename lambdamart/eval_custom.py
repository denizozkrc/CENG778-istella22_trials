import pandas as pd
import ir_measures
from ir_measures import *
#from rankeval.dataset import Dataset
#from rankeval.model import RTEnsemble
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file

def lambdaMart():
    # LambdaMART, MonoT5
    # Remember to create test.monoT5.svm as specified in readme !!!
    TEST_FILE="data/test.monoT5.svm"
    MODEL_FILE="models/lambdamart.monoT5.lgb"

    X_test, y_test, q_test = load_svmlight_file(TEST_FILE, query_id=True)
    lgbm_lmart = lgb.Booster(model_file=MODEL_FILE)
    predictions = lgbm_lmart.predict(X_test)

    #predictions = lgbm_lmart.score(test_subset, detailed=False)
    print(predictions[0:5])
    y = y_test
    print(y[0:5])

    return predictions, y_test, q_test
