import argparse
import time
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgbm
from copy import deepcopy
from functools import reduce
from typing import List, Dict
from .utils import copy_repeat, check_create_folder, nan_mask

import gc


class Pipeline:
    
    def __init__(self, data: Dict, feature, target, model, out_name=None):
        
        self.core = {}
        self.data = data
        self.feature = feature    
        
        if type(target) == list and type(model) == list:
            assert len(target) == len(model)
            
        if type(target) == list and type(out_name) == list:
            assert len(target) == len(out_name)
            
            
        self.target = target if type(target) == list else [target]
        target_len = len(self.target)
        self.core['model'] = model if type(model) == list else \
                             copy_repeat(model, target_len)
        if out_name is None:
            self.out_name = ['y_{}'.format(k) for k in range(target_len)]
        if type(out_name) is str:
            self.out_name = [out_name]
        if type(out_name) == list:
            self.out_name = out_name
        

    def fit(self, index: List[str], metric=None, target_filter_foo=nan_mask):
        
        if type(metric) == list:
            assert len(self.target) == len(metric)
        
        if type(target_filter_foo) == list:
            assert len(self.target) == len(target_filter_foo)
            
        metric = metric if type(metric) == list \
                             else [metric] * len(self.target)

        target_filter_foo = target_filter_foo if type(target_filter_foo) == list \
                             else [target_filter_foo] * len(self.target)

        metrics_result = {}
        X = self.feature.calculate(self.data, index)
        for k, target in enumerate(self.target):
            y = target.calculate(self.data, 
                                 X.index.to_frame(index=False))
            leave_mask = target_filter_foo[k](y['y'].values)

            y_ = y[leave_mask]
            X_ = X[leave_mask]
            self.core['model'][k].fit(X_, y_['y'])
            
            if metric[0] is not None:
                pred = self.core['model'][k].predict(X_)
                metric_name = 'metric_{}'.format(self.out_name[k])
                metrics_result[metric_name] = metric[k](y_['y'].values, pred)
            
        return metrics_result


    def execute(self, index):
         
        result = pd.DataFrame()
        X = self.feature.calculate(self.data, index)
        for k, target in enumerate(self.target):
            pred = self.core['model'][k].predict(X)
            result[self.out_name[k]] = pred
        result.index = X.index

        return result


    def export_core(self, path=None):
          
        if path is None:
            now = time.strftime("%d.%m.%y_%H:%M", time.localtime(time.time()))
            path = 'models_data/pipeline_{}'.format(now)

        check_create_folder(path)
        with open('{}.pickle'.format(path), 'wb') as f:
            pickle.dump(self.core, f)


    def load_core(self, path):
        
        with open(path, 'rb') as f:
            self.core = pickle.load(f)



class MergePipeline:
    
    def __init__(self, pipeline_list:List, execute_merge_on):
        
        self.pipeline_list = pipeline_list
        self.execute_merge_on = execute_merge_on


    def fit(self, index):
        
        for pipeline in self.pipeline_list:
            pipeline.fit(index)


    def _single_batch(self, batch):
        dfs = []
        for pipeline in self.pipeline_list:
            dfs.append(pipeline.execute(batch))
            
        batch_result = reduce(lambda l, r: pd.merge(
            l, r, on=self.execute_merge_on, how='left'), dfs)

        return batch_result


    def execute(self, index, batch_size=None) -> pd.DataFrame:
       
        if batch_size is None:
            batch_size = len(index)
        batches = [index[k:k+batch_size] 
                    for k in range(0, len(index), batch_size)]
        result = []
        for batch in batches:
            result.append(self._single_batch(batch))
           
        result = pd.concat(result, axis=0)

        return result
            
            
class LoadingPipeline:
    
    def __init__(self, data_loader, columns:List[str]):
        
        self.data_loader = data_loader
        self.columns = columns
       
    def fit(self, index):
        None

    def execute(self, index):
         
        data_df = self.data_loader.load(index)
        data_df = data_df[self.columns]
        return data_df            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


