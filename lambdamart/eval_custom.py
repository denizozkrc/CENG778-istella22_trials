import pandas as pd
import ir_measures
from ir_measures import *
from rankeval.dataset import Dataset
from rankeval.model import RTEnsemble

def lambdaMart():
    # LambdaMART, MonoT5
    # Remember to create test.monoT5.svm as specified in readme !!!
    TEST_FILE="data/test.monoT5.svm"
    MODEL_FILE="models/lambdamart.monoT5.lgb"

    test_subset = Dataset.load(TEST_FILE, format="svmlight", name="Istella-test")
    print("Istella - Test Set")
    print("Num. features: ", test_subset.n_features)
    print("Num. queries: ", test_subset.n_queries)
    print("Num. instances: ", test_subset.n_instances)

    lgbm_lmart = RTEnsemble(MODEL_FILE, name="LGBM_lmart", format="LightGBM")
    print("Model statistics")
    print("Num. Trees: ", lgbm_lmart.n_trees)

    y_pred = lgbm_lmart.score(test_subset, detailed=False)
    print(y_pred[0:5])
    y = test_subset.y
    print(y[0:5])

    qid = 0
    total_counter = 0
    run = []
    qrels = []
    while (qid < len(test_subset.get_query_sizes())):
        doc_id = 0
        while (doc_id < test_subset.get_query_sizes()[qid]):
            run.append(ir_measures.ScoredDoc(str(qid), str(doc_id), float(y_pred[total_counter])))
            qrels.append(ir_measures.Qrel(str(qid), str(doc_id), int(y[total_counter])))
            doc_id += 1
            total_counter += 1
        qid += 1

    measures_dict = ir_measures.calc_aggregate([AP, RR, P(rel=1)@1, P(rel=1)@5, P(rel=1)@10, R@100, R@1000, nDCG(dcg='exp-log2')@5, nDCG(dcg='exp-log2')@10, nDCG(dcg='exp-log2')@20, Judged@10], qrels, run)
    measures_df = pd.DataFrame.from_dict(measures_dict, orient='index')

    return run, qrels, measures_df
