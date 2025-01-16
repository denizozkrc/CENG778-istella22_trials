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
monoT5_res_tu_df = pd.DataFrame(monoT5_res_tu)

"""monoT5_res_tut['score'] = normaliseMonoScores(monoT5_res_tut)
pt.io.write_results(monoT5_res_tut, 'norm_runs/monot5.titleurltext.normalized.res.gz')
monoT5_res_tut_df = pd.DataFrame(monoT5_res_tut)"""


# LambdaMART results
run, qrels, measures_df = lambdaMart()

run_df = pd.DataFrame([{'qid': sd.query_id, 'docid': sd.doc_id, 'score': sd.score} for sd in run])
run_df['score'] = normaliseMonoScores(run_df)
pt.io.write_results(run_df, 'norm_runs/lambdamart.normalized.res.gz')

qrels_df = pd.DataFrame([{'qid': q.query_id, 'docid': q.doc_id, 'rel': q.relevance} for q in qrels])
pt.io.write_results(qrels_df, 'norm_runs/lambdamart.normalized.qrels.gz')


# NeuralRanker results
grouped_run = run_df.groupby('qid')
grouped_mono = monoT5_res_tu_df.groupby('qid')
grouped_qrels = qrels_df.groupby('qid')

unique_qids_lambda = run_df['qid'].unique()
unique_qids_mono = monoT5_res_tu_df['qid'].unique()

split_idx_lambda = len(unique_qids_lambda) // 2
train_qids_lambda = unique_qids_lambda[:split_idx_lambda]
test_qids_lambda = unique_qids_lambda[split_idx_lambda:]

split_idx_mono = len(unique_qids_mono) // 2
train_qids_mono = unique_qids_mono[:split_idx_mono]
test_qids_mono = unique_qids_mono[split_idx_mono:]

train_run_df = run_df[run_df['qid'].isin(train_qids_lambda)]
test_run_df = run_df[run_df['qid'].isin(test_qids_lambda)]

train_qrels_df = qrels_df[qrels_df['qid'].isin(train_qids_lambda)]
test_qrels_df = qrels_df[qrels_df['qid'].isin(test_qids_lambda)]

train_mono_df = monoT5_res_tu_df[monoT5_res_tu_df['qid'].isin(train_qids_mono)]
test_mono_df = monoT5_res_tu_df[monoT5_res_tu_df['qid'].isin(test_qids_mono)]

# training
#y = np.array(normaliseRankScores(qrels_df['rel']))
y_train = np.array(normaliseRankScores(train_qrels_df['rel']))
# when we add another reranker, must add its rank to here somehow. Average maybe?

X_train = np.vstack((train_mono_df['score'], train_run_df['score'])).T  # (n_samples, 2)

LR = LinearRegressor()
LR.fit(X_train, y_train)

w1, w2 = LR.model.coef_
print(f"Learned Weights: w1={w1}, w2={w2}")

# testing
#X_test = np.vstack((test_mono_df['score'], test_run_df['score'])).T  # (n_samples, 2)
combined_test_df = test_run_df.merge(
    test_mono_df[['qid', 'docid', 'score']], 
    on=['qid', 'docid'], 
    how='left', 
    suffixes=('_run', '_mono')
)
combined_test_df['score_mono'] = combined_test_df['score_mono'].fillna(0)
X_test = np.vstack((combined_test_df['score_mono'], combined_test_df['score_run'])).T  # (n_samples, 2)

y_pred = LR.predict(X_test)

# Evaluate
# Rank the documents within each query
combined_test_df['score'] = y_pred
combined_test_df['rank'] = combined_test_df.groupby('qid')['score'].rank(method='first', ascending=False).astype(int)
ranked_results = combined_test_df.sort_values(by=['qid', 'rank'])

"""ranked_results = test_run_df.copy()
ranked_results['score'] = y_pred
ranked_results['rank'] = ranked_results.groupby('qid')['score'].rank(method='first', ascending=False).astype(int)
"""

# Write the ranked results to a file
pt.io.write_results(ranked_results, 'norm_runs/neuralranker.ranked.res.gz')

# Evaluate the ranked results
pt.Experiment(
    [ranked_results],  # Use a PyTerrier pipeline or results DataFrame
    [test_qrels_df],   # Ground-truth relevance judgments
    measures=[P, nDCG, RR, AP],  # Specify evaluation measures
    perquery=True
)

# Evaluate the initial results

