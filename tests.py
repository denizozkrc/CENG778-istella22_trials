import os
import ir_datasets
import pyterrier as pt
from pyterrier.measures import P, nDCG, RR, AP
from lambdamart.eval_custom import lambdaMart
from secondReRanker import LinearRegressor
import pandas as pd

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def seeFormat(res=pt.io.read_results('monoT5/runs/initial.res.gz')):
    print(res.head(10)[['qid', 'docno', 'score', 'rank']])
    # print(res.columns)


# MonoT5 results
# monoT5_res_tut = pt.io.read_results('monoT5/runs/monot5.titleurltext.res.gz')
monoT5_res_tu = pt.io.read_results('monoT5/runs/monot5.titleurl.res.gz')

#monoT5_res_tu['score'] = normaliseMonoScores(monoT5_res_tu)
monoT5_res_tu_df = pd.DataFrame(monoT5_res_tu)

qid_to_remove = {'11904', '5184', '12774', '16467', '16118', '13241'}

monoT5_res_tu_df = monoT5_res_tu_df[~monoT5_res_tu_df['qid'].isin(qid_to_remove)]

monoT5_res_tu_df["qid"] = monoT5_res_tu_df["qid"].astype(int)
monoT5_res_tu_df["score"] = monoT5_res_tu_df["score"].astype(float)
monoT5_res_tu_df["rank"] = monoT5_res_tu_df["rank"].astype(int)

monoT5_res_tu_df_ordered = monoT5_res_tu_df.sort_values(by=["qid", "docno"])
monoT5_res_tu_df_ordered = monoT5_res_tu_df_ordered.reset_index(drop=True)

print(monoT5_res_tu_df_ordered)


"""monoT5_res_tut['score'] = normaliseMonoScores(monoT5_res_tut)
pt.io.write_results(monoT5_res_tut, 'norm_runs/monot5.titleurltext.normalized.res.gz')
monoT5_res_tut_df = pd.DataFrame(monoT5_res_tut)"""


# LambdaMART results
# run, qrels, measures_df = lambdaMart()
predictions, y_test, q_test = lambdaMart()


# NeuralRanker results
unq_qids = np.unique(q_test)
combined_predictions = [[] for _ in range(len(unq_qids))]
true_scores = [[] for _ in range(len(unq_qids))]
index = 0
for i in range(len(q_test)):
    combined_predictions[index] = np.append(combined_predictions[index], [[monoT5_res_tu_df["score"][i], predictions[i]]])
    true_scores[index] = np.append(true_scores[index], y_test[i])
    if i < len(q_test)-1 and q_test[i+1] != q_test[i]:
        index += 1

split_indices = len(unq_qids) // 2

combined_predictions_train = combined_predictions[:split_indices]
combined_predictions_test = combined_predictions[split_indices:]
true_scores_train = true_scores[:split_indices]
true_scores_test = true_scores[split_indices:]

border_qid = unq_qids[split_indices]
first_test_index = q_test.tolist().index(border_qid)

# training

y_train = np.array(true_scores_train).flatten()
# when we add another reranker, must add its rank to here somehow. Average maybe?

X_train = np.array(combined_predictions_train).reshape(-1, 2)

LR = LinearRegressor()
LR.fit(X_train, y_train)

w1, w2 = LR.model.coef_
print(f"Learned Weights: w1={w1}, w2={w2}")

X_test = np.array(combined_predictions_test).reshape(-1, 2)
# (n_samples, 2)

y_pred = LR.predict(X_test).flatten()

y_pred_df = pd.DataFrame({
    "qid": q_test[first_test_index:],
    "docno": monoT5_res_tu_df["docno"][first_test_index:],
    "score": y_pred   # Predicted scores
})
true_scores_test_df = pd.DataFrame({
    "qid": q_test[first_test_index:],
    "docno": monoT5_res_tu_df["docno"][first_test_index:],
    "score": true_scores_test   # True scores
})

# Evaluate the ranked results
pt.Experiment(
    [y_pred_df],  
    [true_scores_test_df],   
    measures=[P, nDCG, RR, AP],
    perquery=True
)

# Evaluate the initial results