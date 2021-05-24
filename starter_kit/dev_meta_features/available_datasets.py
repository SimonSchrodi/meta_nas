from pathlib import Path
import yaml
import os
import json

def get_portfolio(filepath):
  return json.load(open(filepath,"r"))['portfolio']

# 'binary_alpha_digits' # augmentation failure
all_datasets = ['cifar100', 'cycle_gan_vangogh2photo', 'uc_merced', 'cifar10', 'cmaterdb_devanagari', 
                'cmaterdb_bangla', 'mnist', 'horses_or_humans', 'kmnist', 'cycle_gan_horse2zebra', 'cycle_gan_facades', 
                'cycle_gan_apple2orange', 'imagenet_resized_32x32', 'cycle_gan_maps', 'omniglot', 'imagenette', 'emnist_byclass', 
                'svhn_cropped', 'colorectal_histology', 'coil100', 'stanford_dogs', 'rock_paper_scissors', 'tf_flowers', 
                'cycle_gan_ukiyoe2photo', 'cassava', 'fashion_mnist', 'emnist_mnist', 'cmaterdb_telugu', 'malaria', 'eurosat_rgb', 
                'emnist_balanced', 'cars196', 'cycle_gan_iphone2dslr_flower', 'cycle_gan_summer2winter_yosemite', 'cats_vs_dogs']

nascomp_dev = ['devel_dataset_0', 'devel_dataset_1', 'devel_dataset_2']
_4_resnet34_cosine = "dev_meta_features/task2vec_results/meta_dataset_creation_resnet34/cosine/metadataset_info/info.json"
_10r_resnet34_cosine = "dev_meta_features/task2vec_results/meta_dataset_creation_resnet34/cosine_with_curation_10r/metadataset_info/info.json"
_50r_resnet34_cosine = "dev_meta_features/task2vec_results/meta_dataset_creation_resnet34/cosine_with_curation_50r/metadataset_info/info.json"
_3_resnet34_cosine = "dev_meta_features/task2vec_results/meta_dataset_creation_resnet34/cosine_with_curation_3/metadataset_info/info.json"
_10_resnet34_cosine = "dev_meta_features/task2vec_results/meta_dataset_creation_resnet34/cosine_with_curation_10/metadataset_info/info.json"
_50_resnet34_cosine = "dev_meta_features/task2vec_results/meta_dataset_creation_resnet34/cosine_with_curation_50/metadataset_info/info.json"
_nosim_resnet34_cosine = "dev_meta_features/task2vec_results/meta_dataset_creation_resnet34/cosine_with_curation_no_similarity/metadataset_info/info.json"
#nascomp_portfolio = get_portfolio(_50_resnet34_cosine)

_4_resnet34_cosine_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_4'
_10r_resnet34_cosine_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_with_curation_10r'
_50r_resnet34_cosine_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_with_curation_50r'
_3_resnet34_cosine_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_with_curation_3'
_10_resnet34_cosine_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_with_curation_10'
_50_resnet34_cosine_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_with_curation_50'
_nosim_resnet34_cosine_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_with_curation_no_similarity'

GROUPS = {
          'all': all_datasets, 
          'nascomp_dev': nascomp_dev
         }

AUTODL_MAIN_DIR = '/data/aad/image_datasets/augmented_datasets'

if __name__ == '__main__':

    for group, elements in GROUPS.items():
      print('Dataset group {} contains {} dataset(s). These are ->'.format(group, len(elements)))
      print('\n'.join(elements))
      print('='*60)
