import os
import ir_datasets
import pyterrier as pt
from pyterrier.measures import P, nDCG, RR, AP
from monoT5.run_monot5 import run_monoT5


def seeFormat(res=pt.io.read_results('monoT5/runs/initial.res.gz')):
    print(res.head(10)[['qid', 'docno', 'score', 'rank']])
    #print(res.columns)


def normaliseMonoScores(res):
    normalized_scores = (res['score'] - res['score'].min()) / (res['score'].max() - res['score'].min())
    return normalized_scores


run_monoT5()
monoT5_res_tut = pt.io.read_results('monoT5/runs/monot5.titleurltext.res.gz')
monoT5_res_tu = pt.io.read_results('monoT5/runs/monot5.titleurl.res.gz')

monoT5_res_tu['score'] = normaliseMonoScores(monoT5_res_tu)
pt.io.write_results(monoT5_res_tu, 'norm_runs/monot5.titleurl.normalized.res.gz')

monoT5_res_tut['score'] = normaliseMonoScores(monoT5_res_tut)
pt.io.write_results(monoT5_res_tut, 'norm_runs/monot5.titleurltext.normalized.res.gz')

#seeFormat(monoT5_res_tu)
#seeFormat(monoT5_res_tut)
