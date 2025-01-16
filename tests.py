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


def normaliseMonoScores(res):
    normalized_scores = (res['score'] - res['score'].min()) / (res['score'].max() - res['score'].min())
    return normalized_scores

def normaliseRankScores(rel):
    return 1-(rel - rel.min()) / (rel.max() - rel.min())


# MonoT5 results

monoT5_res_tut = pt.io.read_results('monoT5/runs/monot5.titleurltext.res.gz')
monoT5_res_tu = pt.io.read_results('monoT5/runs/monot5.titleurl.res.gz')

monoT5_res_tu['score'] = normaliseMonoScores(monoT5_res_tu)
pt.io.write_results(monoT5_res_tu, 'norm_runs/monot5.titleurl.normalized.res.gz')

monoT5_res_tut['score'] = normaliseMonoScores(monoT5_res_tut)
pt.io.write_results(monoT5_res_tut, 'norm_runs/monot5.titleurltext.normalized.res.gz')


# LambdaMART results

run, qrels, measures_df = lambdaMart()

run_df = pd.DataFrame([{'qid': sd.query_id, 'docid': sd.doc_id, 'score': sd.score} for sd in run])
run_df['score'] = normaliseMonoScores(run_df)
pt.io.write_results(run_df, 'norm_runs/lambdamart.normalized.res.gz')

qrels_df = pd.DataFrame([{'qid': q.query_id, 'docid': q.doc_id, 'rel': q.relevance} for q in qrels])
pt.io.write_results(qrels_df, 'norm_runs/lambdamart.normalized.qrels.gz')


# NeuralRanker results

mono_res_tu = np.array(monoT5_res_tu['score'])
#mono_res_tut = np.array(monoT5_res_tut['score'])
lamb_res = np.array(run_df['score'])
y = np.array(normaliseRankScores(qrels_df['rel']))

X = np.vstack((mono_res_tu, lamb_res)).T  # (n_samples, 2)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

LR = LinearRegressor()
LR.fit(X, y)

w1, w2 = LR.model.coef_
print(f"Learned Weights: w1={w1}, w2={w2}")

# Predict using learned weights
y_pred = LR.predict(X)

final_scores = w1 * X[:, 0] + w2 * X[:, 1]
ranked_indices = np.argsort(final_scores)[::-1]  # Sort in descending order

run_df['score'] = final_scores
pt.io.write_results(run_df, 'norm_runs/neuralranker.normalized.res.gz')

qrels_df = qrels_df.merge(run_df[['qid', 'docid', 'score']], on=['qid', 'docid'], how='left')
qrels_df = qrels_df.fillna(0)
qrels_df['ranked'] = qrels_df.groupby('qid')['score'].rank(method='first', ascending=False).astype(int)
pt.io.write_results(qrels_df, 'norm_runs/neuralranker.normalized.qrels.gz')
