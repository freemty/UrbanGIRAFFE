import imp
import os
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.evaluators.eval_fid import Evaluator as fidEvaluator
from lib.evaluators.eval_miou import Evaluator as miouEvaluator
# from lib.evaluators.eval_pq import 


evaluator_dict = {
    'fid': fidEvaluator,
    'miou': miouEvaluator
}


def _evaluator_factory(cfg):
    evaluator = evaluator_dict[cfg.evaluator_type]()
    return evaluator

def make_evaluator(cfg):
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg)