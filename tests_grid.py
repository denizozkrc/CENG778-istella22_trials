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

grid = {'w_1': [0.1, 0.2, 0.3, 0.4, 0.5]}


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

#res = pt.io.read_results('monoT5/runs/initial.res.gz')
#print(len(monoT5_res_tu_df['qid'].unique()))

# LambdaMART results
run, qrels, measures_df = lambdaMart()

run_df = pd.DataFrame([{'qid': sd.query_id, 'docid': sd.doc_id, 'score': sd.score} for sd in run])
run_df['score'] = normaliseMonoScores(run_df)
pt.io.write_results(run_df, 'norm_runs/lambdamart.normalized.res.gz')

qrels_df = pd.DataFrame([{'qid': q.query_id, 'docid': q.doc_id, 'rel': q.relevance} for q in qrels])
qrels_df['rel'] = normaliseRankScores(qrels_df['rel'])
pt.io.write_results(qrels_df, 'norm_runs/lambdamart.normalized.qrels.gz')


# GridSearch for weights
"""grouped_run = run_df.groupby('qid')
grouped_mono = monoT5_res_tu_df.groupby('qid')
grouped_qrels = qrels_df.groupby('qid')"""

combined_df = run_df.merge(
    monoT5_res_tu_df[['qid', 'docid', 'score']], 
    on=['qid', 'docid'], 
    how='left', 
    suffixes=('_run', '_mono')
).assign(score_mono=lambda x: x['score_mono'].fillna(0))
#combined_df['score_mono'] = combined_df['score_mono'].fillna(0)

combined_df = combined_df.merge(
    qrels_df[['qid', 'docid', 'rel']], 
    on=['qid', 'docid'], 
    how='left'
).assign(rel=lambda x: x['rel'].fillna(0))
#combined_df['rel'] = combined_df['rel'].fillna(0)


X = combined_df[['score_run', 'score_mono']].copy()
y = combined_df['rel']

min_mse = np.inf
best_w_1 = None
for w_1 in grid['w_1']:
    X['score'] = w_1 * X['score_run'] + (1 - w_1) * X['score_mono']
    mse = mean_squared_error(y, X['score'])
    if mse < min_mse:
        min_mse = mse
        best_w_1 = w_1

print(f'Best w_1: {best_w_1}, MSE: {min_mse}')

combined_df['score'] = best_w_1 * combined_df['score_run'] + (1 - best_w_1) * combined_df['score_mono']
#rank
combined_df = combined_df.sort_values(by=['qid', 'score'], ascending=[True, False])
combined_df['rank'] = combined_df.groupby('qid').cumcount() + 1

pt.io.write_results(combined_df[['qid', 'docid', 'score', 'rank', 'rel']], 'norm_runs/combined.normalized.res.gz')
