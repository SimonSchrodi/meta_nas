import os
import sys
import pickle
import json
import math
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import random

sys.path.append(os.getcwd())
from available_datasets import all_datasets, get_portfolio, AUTODL_MAIN_DIR

import tensorflow as tf
tf.enable_eager_execution()

def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO score.py: <message>
  """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger

verbosity_level = "INFO"
logger = get_logger(verbosity_level)

def get_autodl_dataset(dataset_dir):

    from competition import data_io
    from competition.dataset import AutoDLDataset  # THE class of AutoDL datasets
    from competition.score import get_solution

    ##################################################
    #### COPIED FROM THE AUTODL INGESTION PROGRAM ####
    ##################################################

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(dataset_dir)
    #### Delete zip files and metadata file
    datanames = [x for x in datanames if x.endswith(".data")]

    if len(datanames) != 1:
        raise ValueError(
            "{} datasets found in dataset_dir={}!\n".format(len(datanames), dataset_dir) +
            "Please put only ONE dataset under dataset_dir."
        )

    basename = datanames[0]
    
    ##### Begin creating training set and test set #####
    D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(dataset_dir, basename, "test"))

    logger.info("************************************************")
    logger.info("******** Processing dataset " + basename[:-5].capitalize() + " ********")
    logger.info("************************************************")

    logger.info('Length of the training set: %d' % D_train.get_metadata().size())
    logger.info('Length of the test set: %d' % D_test.get_metadata().size())

    return D_train, D_test, get_solution(dataset_dir)

def get_dataset_hwc(train_dataset, test_dataset):
    metadata_train = train_dataset.get_metadata()
    metadata_test = test_dataset.get_metadata()
    h_train, w_train, c_train = metadata_train.get_tensor_size()
    h_test, w_test, c_test = metadata_test.get_tensor_size()

    h = -1 if h_train == -1 or h_test == -1 else max(h_train, h_test)
    w = -1 if w_train == -1 or w_test == -1 else max(w_train, w_test)
    c = max(c_train, c_test)

    return h, w, c

def pad_images(images, max_h, max_w):

    padded_images = []
    for img in images:
        img = torch.from_numpy(img)
        _, img_h, img_w = img.size()  
        w_pad_1 = math.ceil((max_w-img_w)/2)
        w_pad_2 = math.floor((max_w-img_w)/2)
        h_pad_1 = math.ceil((max_h-img_h)/2)
        h_pad_2 = math.floor((max_h-img_h)/2)
        padder = torch.nn.ZeroPad2d((w_pad_1, w_pad_2, h_pad_1, h_pad_2))
        padded_images.append(padder(img).numpy())

    return padded_images

def interpolate_images(images, max_h, max_w):

    int_images = []
    for img in images:
        img = torch.from_numpy(img).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size = (max_h, max_w), mode = 'bilinear')
        int_images.append(img.squeeze(0).numpy())

    return int_images

def autodl_to_torch(dataset):

    _, _, c = dataset.get_metadata().get_tensor_size()
    size_limit = 224 # max size 
    
    images = []
    labels = []
    for i, (image, label) in enumerate(dataset.get_dataset().take(-1)):

        image = image.numpy()
        label = label.numpy()

        if c == 1:
            image = np.stack((image.squeeze(-1),)*3, axis=-1)
        images.append(image.squeeze(0).transpose(2, 0, 1))
        labels.append(np.argmax(label))

    if len(images) > 45000:
        pairs = random.sample(list(zip(images, labels)), 45000)
        images, labels = zip(*pairs)

    num_classes=max(labels)+1

    images = interpolate_images(images, size_limit, size_limit)

    images = torch.from_numpy(np.array(images))
    labels = torch.from_numpy(np.array(labels)).long()

    dataset = torch.utils.data.TensorDataset(images, labels)

    return dataset, num_classes

def get_nascomp_dataset(dataset_dir):
    train_x = np.load(os.path.join(dataset_dir,'train_x.npy'))
    train_y = np.load(os.path.join(dataset_dir,'train_y.npy'))
    valid_x = np.load(os.path.join(dataset_dir,'valid_x.npy'))
    valid_y = np.load(os.path.join(dataset_dir,'valid_y.npy'))
    test_x = np.load(os.path.join(dataset_dir,'test_x.npy'))
    test_y = np.load(os.path.join(dataset_dir,'test_y.npy'))

    return (train_x, train_y), \
           (valid_x, valid_y), \
           (test_x, test_y)

def nascomp_to_torch(dataset):
    # nascomp dataset image shape: N x C x W x H
    # autodl input image shape: N x H x W x C
    # task2vec input image shape: N x C x H x W
    images, labels = dataset # nascomp
    c = images.shape[1] 
    
    formatted_images = []
    for image in images:
        if c == 1:
            image = np.stack((image, )*3, axis=-1)
            image = image.squeeze(0).transpose(2, 1, 0)
        formatted_images.append(image)

    size_limit = 224
    images = interpolate_images(formatted_images, size_limit, size_limit)

    images = torch.from_numpy(np.array(images)) # task2vec
    labels = torch.from_numpy(labels).long()

    dataset = torch.utils.data.TensorDataset(images, labels)
    num_classes=max(labels)+1
    
    return dataset, num_classes.item()

def tensorflow2numpy(dataset, num_channels, test_solution = None, **opts):

    size_limit = opts['size_limit'] if 'size_limit' in opts else 224
    h, w = opts['shape'] if 'shape' in opts else -1, -1

    images = []
    labels = []
    for i, (image, label) in enumerate(dataset.get_dataset().take(-1)):

        image = image.numpy()
        if test_solution is not None:
            label = test_solution[i]

        images.append(image.squeeze(0).transpose(2, 1, 0))
        labels.append(np.argmax(label))
    
    # Bring the images into the same size or trim them if too large to avoid memory issues
    if h == -1 or w == -1 or h > size_limit or w > size_limit:
        max_w = min(max([img.shape[1]for img in images]), size_limit)
        max_h = min(max([img.shape[2]for img in images]), size_limit)
        images = interpolate_images(images, max_w, max_h)

    X = np.array(images)
    y = np.array(labels)

    return X, y
    

def autodl2nascomp(train_dataset, test_dataset, test_solution):
    
    from sklearn.model_selection import train_test_split

    h, w, c = get_dataset_hwc(train_dataset, test_dataset)
    size_limit = 224 # max size 
    
    X, y = tensorflow2numpy(train_dataset, c, size_limit = size_limit, shape = (h, w))
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size = 0.1, random_state = 42, stratify = y)
    test_x, test_y = tensorflow2numpy(test_dataset, c, test_solution, size_limit = size_limit, shape = (h, w))

    return (train_x, train_y), \
           (valid_x, valid_y), \
           (test_x, test_y)

def convert_metadata_to_df(metadata):
    k, v = list(metadata.items())[0]
    columns = sorted(v.keys())
    columns_edited = False

    features_lists = []
    indices = []

    for key, values_dict in sorted(metadata.items()):
        indices.append(key)
        feature_list = [values_dict[k] for k in sorted(values_dict.keys())]

        # below loop flattens feature list since there are tuples in it &
        # it extends columns list accordingly
        for i, element in enumerate(feature_list):
            if type(element) is tuple:
                # convert tuple to single list elements
                slce = slice(i, i + len(element) - 1)

                feature_list[slce] = list(element)

                if not columns_edited:
                    columns_that_are_tuples = columns[i]
                    new_columns = [
                        columns_that_are_tuples + "_" + str(i) for i in range(len(element))
                    ]
                    columns[slce] = new_columns
                    columns_edited = True

        features_lists.append(feature_list)

    return pd.DataFrame(features_lists, columns=columns, index=indices)


def dump_meta_features_df_and_csv(meta_features, n_augmentations, output_path, file_name="meta_features", samples_along_rows=False, n_samples=None):

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        
    if not isinstance(meta_features, pd.DataFrame):
        df = convert_metadata_to_df(meta_features)
    else:
        df = meta_features

    df.to_csv((output_path / file_name).with_suffix(".csv"))
    df.to_pickle((output_path / file_name).with_suffix(".pkl"))

    print("meta features data dumped to: {}".format(output_path))

def create_nascomp_datasets(source_datasets_dir, portfolio, target_datasets_dir):
    train_sizes = []
    test_sizes = []
    for name in portfolio:
        logger.info(f"Converting {name}")
        dataset_dir = os.path.join(source_datasets_dir, *name.split('-'))
        
        train_dataset, test_dataset, test_solution = get_autodl_dataset(dataset_dir)
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = autodl2nascomp(train_dataset, test_dataset, test_solution)

        print('Training set shape:', train_x.shape)
        print('Validation set shape:',valid_x.shape)
        print('Test set shape:', test_x.shape)

        train_sizes.append(train_x.shape[0])
        test_sizes.append(test_x.shape[0])

        name = name +'_dataset'
        target_dir = os.path.join(target_datasets_dir, name)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        n_classes = train_dataset.get_metadata().get_output_size()
        
        dataset_metadata = {"batch_size": 256, "n_classes": n_classes, "lr": 0.01, "benchmark": 0.0, "name": name}
        json.dump(dataset_metadata, open(os.path.join(target_dir, "dataset_metadata"),"w"))
        
        np.save(os.path.join(target_dir, 'train_x'), train_x)
        np.save(os.path.join(target_dir, 'train_y'), train_y)

        np.save(os.path.join(target_dir, 'valid_x'), valid_x)
        np.save(os.path.join(target_dir, 'valid_y'), valid_y)

        np.save(os.path.join(target_dir, 'test_x'), test_x)
        np.save(os.path.join(target_dir, 'test_y'), test_y)

    for s1, s2 in zip(train_sizes, test_sizes):
        print(s1, s2, s1>1e4 and s2>1e3)

def visualize_nascomp_metadataset(portfolio, datasets_dir, image_output_dir, n = 3):

    for name in portfolio:
        logger.info(f"Visualizing {name}")
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = get_nascomp_dataset(os.path.join(datasets_dir, name)+'_dataset')
        
        subdir = os.path.join(image_output_dir, name)
        if not os.path.isdir(subdir):
            os.makedirs(subdir)

        samples = np.random.choice(range(len(train_x)), n)
        samples = train_x[samples]
        for i, sample in enumerate(samples):
            plt.imshow(sample.transpose(2, 1, 0), interpolation='nearest')
            plt.savefig(os.path.join(subdir, str(i)+'.png'))
            plt.close()

        samples = np.random.choice(range(len(valid_x)), n)
        samples = valid_x[samples]
        for i, sample in enumerate(samples):
            plt.imshow(sample.transpose(2, 1, 0), interpolation='nearest')
            plt.savefig(os.path.join(subdir, str(i+3)+'.png'))
            plt.close()

        samples = np.random.choice(range(len(test_x)), n)
        samples = test_x[samples]
        for i, sample in enumerate(samples):
            plt.imshow(sample.transpose(2, 1, 0), interpolation='nearest')  
            plt.savefig(os.path.join(subdir, str(i+6)+'.png'))
            plt.close()

if __name__ == "__main__":

    from available_datasets import _50_resnet34_cosine_path, _50_resnet34_cosine, _nosim_resnet34_cosine_path, _nosim_resnet34_cosine, _3_resnet34_cosine_path, _3_resnet34_cosine
    
    metadataset_path = _nosim_resnet34_cosine_path
    nascomp_portfolio = get_portfolio(_nosim_resnet34_cosine)
    create_nascomp_datasets(AUTODL_MAIN_DIR, nascomp_portfolio, metadataset_path)
    #visualize_nascomp_metadataset(nascomp_portfolio, metadataset_path, 'meta_dataset_images_cosine_with_curation')

    os.system('cp -r public_data_12-03-2021_13-33/* '+metadataset_path)
    '''
    (train_x, train_y), \
    (valid_x, valid_y), \
    (test_x, test_y) = get_nascomp_dataset('/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_small/16-omniglot_dataset')

    print(train_x.shape)
    print(valid_x.shape)
    print(test_x.shape)

    print(len(np.unique(train_y)))
    print(len(np.unique(valid_y)))
    print(len(np.unique(test_y)))
    '''

