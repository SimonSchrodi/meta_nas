import numpy as np
import torch
import torch.nn as nn
import torchvision

import nas_helpers
from models import *
import train_helpers

import time
import operator

"""
Your submission must contain a class NAS that has a `search` method. The arguments
we'll pass and the output we'll expect are detailed below, but beyond that you can
do whatever you want within the class. Feel free to add other python files that are
imported, as long as you bundle them into the submission zip and they are imported when
`import nas` is called.

For information, you can check out the ingestion program to see exactly how we'll interact 
with this class. To submit, zip this file and any helpers files together with the dataset_metadata file
 and upload it to the CodaLab page.
"""


class NAS:
    def __init__(self):
        self.n_epochs = 64
        self.search_duration =  14400 - 600 # 4h - slack (10min)
        self.meta_learner = 'timm'
        # for debugging purposes of a single default model
        self.return_default = False
        self.default_model = 'ResNet18'
        
        if self.meta_learner == 'iterate':
            self.models = light_portfolio
        elif self.meta_learner == 'timm':
            self.models = timm_portfolio

    """
	search() Inputs:
		train_x: numpy array of shape (n_datapoints, channels, weight, height)
		train_y: numpy array of shape (n_datapoints)
		valid_x: numpy array of shape (n_datapoints, channels, weight, height)
		valid_y: numpy array of shape (n_datapoints)
		dataset_metadata: dict, contains:
			* batch_size: the batch size that will be used to train this da==taset
			* n_classes: the total number of classes in the classification task
			* lr: the learning rate that will be used to train this dataset
			* benchmark: the threshold used to determine zero point of scoring; your score on the dataset will equal
			 '10 * (test_acc - benchmark) / (100-benchmark)'. 
                - This means you can score a maximum of 10 points on each dataset: a full 10 points will be awarded for 100% test accuracy, 
                while 0 points will be awarded for a test accuracy equal to the benchmark. 
			* name: a unique name for this dataset
					
	search() Output:
		model: a valid PyTorch model
    """

    def search(self, train_x, train_y, valid_x, valid_y, metadata):
        search_time_limit = time.time() + self.search_duration

        self._prepare_train(train_x, train_y, valid_x, valid_y, metadata)

        self.performance_stats = {}
        n = 0
        inc = -1
        prev_train_duration = 0
        while time.time() < search_time_limit and not self.return_default:
            # choose, load and train model
            key, model = self._meta_learner(n, num_classes=metadata['n_classes'])
            
            print('training', key)
            try:
                train_start_time = time.time()
                res = self._train(model, search_time_limit)                
                train_duration = time.time() - train_start_time
                self.performance_stats[key] = (res, train_duration)
                if res > inc:
                    search_time_limit = search_time_limit + prev_train_duration - train_duration # update time
                    prev_train_duration = train_duration
                    inc = res

                print(key, 'finished', res)
            except Exception as e:
                print(key, 'failed', e)
                self.performance_stats[key] = -1
            n += 1

        print('performance stats:', self.performance_stats)

        self.performance_stats = {k:v[0] for k,v in self.performance_stats.items()} # filter out runtime information
        key = max(self.performance_stats.items(), key=operator.itemgetter(1))[0] if self.performance_stats else self.default_model
        model = self.models[key]()
        return nas_helpers.reshape_model(model=model, channels=self.channels, n_classes=self.n_classes)

    def _meta_learner(self, n, num_classes):
        if self.meta_learner == 'iterate':
            key = list(light_portfolio.keys())[n]
            return key, self.models[key]()  

        elif self.meta_learner == 'timm':            
            # there are 498 models total in timm.list_models()
            key = list(timm_portfolio.keys())[n]
            return key, self.models[key]()

        else:
            raise NotImplementedError
    
    def _prepare_train(self, train_x, train_y, valid_x, valid_y, metadata):
        # load + package data
        self.channels = train_x.shape[1]
        self.n_classes = metadata['n_classes']
        self.batch_size = metadata['batch_size']
        self.lr = metadata['lr']
        self.train_pack = list(zip(train_x, train_y))
        self.valid_pack = list(zip(valid_x, valid_y))

    
    def _train(self, model, train_time_limit):
        # reshape it to this dataset and reset model
        model = nas_helpers.reshape_model(model=model, channels=self.channels, n_classes=self.n_classes)
        train_helpers.reset_weights(model)

        # create data loader
        train_loader = torch.utils.data.DataLoader(self.train_pack, int(self.batch_size), shuffle=False)
        valid_loader = torch.utils.data.DataLoader(self.valid_pack, int(self.batch_size))

        # train
        results = train_helpers.full_training(
            model=model,
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu"),
            lr=self.lr,
            epochs=self.n_epochs,
            train_loader=train_loader,
            valid_loader=valid_loader,
            time_limit=train_time_limit
        )

        return results
