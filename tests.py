import os
import ir_datasets
import pyterrier as pt
from pyterrier.measures import P, nDCG, RR, AP
from lambdamart.eval_custom import lambdaMart
import pandas as pd



def seeFormat(res=pt.io.read_results('monoT5/runs/initial.res.gz')):
    print(res.head(10)[['qid', 'docno', 'score', 'rank']])
    #print(res.columns)


def normaliseMonoScores(res):
    normalized_scores = (res['score'] - res['score'].min()) / (res['score'].max() - res['score'].min())
    return normalized_scores


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
