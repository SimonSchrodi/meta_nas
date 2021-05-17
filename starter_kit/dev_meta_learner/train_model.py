import argparse
import os
import pickle
import timm
from path import Path
import time

from train import torch_evaluator

from utils import get_dataset_paths

import sys
sys.path.append('dev_submission')
import nas_helpers
from models import *

sys.path.append('ingestion_program')
from nascomp.helpers import load_datasets

parser = argparse.ArgumentParser(description='Train models.')
parser.add_argument('--model', type=str, help='Model to train', required=True)
parser.add_argument('--data', type=str, help='Dataset name or path to datasets, if model is not trained on all datasets', required=True)
parser.add_argument('--save_dir', type=str, default=os.getcwd(), help='Dir where to save files')
parser.add_argument('--reshaping', type=str, default='Starter', help='What kind of reshaping to use')


def main():
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.makedirs_p()
    # train model and save results to disc
    for dataset_path in get_dataset_paths(args.data):
        (train_x, train_y), (valid_x, valid_y), test_x, metadata = load_datasets(dataset_path)
        data = (train_x, train_y), (valid_x, valid_y), test_x

        # get model
        assert args.model in timm.list_models()
        model = timm.create_model(args.model, num_classes=metadata['n_classes'])
        model_specific = nas_helpers.reshape_model(model=model, channels=train_x.shape[1], n_classes=metadata['n_classes'], copy_type=args.reshaping)
        try:
            start_time = time.time()
            results = torch_evaluator(model_specific, data, metadata, n_epochs=64, full_train=True)
            results['train_duration'] = time.time() - start_time
            
            _save_dir = save_dir / f'{dataset_path[dataset_path.rfind("/")+1:]}'
            _save_dir = Path(_save_dir)
            _save_dir.makedirs_p()
            save_path = _save_dir / f'{args.model}.pickle'
            with open(save_path, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print(f'{args.model} cannot be trained on {dataset_path}')

if __name__ == "__main__":
    main()