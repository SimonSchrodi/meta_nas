from pathlib import Path
import yaml
import os

# 'binary_alpha_digits' # augmentation failure
all_datasets = ['cifar100', 'cycle_gan_vangogh2photo', 'uc_merced', 'cifar10', 'cmaterdb_devanagari', 
                'cmaterdb_bangla', 'mnist', 'horses_or_humans', 'kmnist', 'cycle_gan_horse2zebra', 'cycle_gan_facades', 
                'cycle_gan_apple2orange', 'imagenet_resized_32x32', 'cycle_gan_maps', 'omniglot', 'imagenette', 'emnist_byclass', 
                'svhn_cropped', 'colorectal_histology', 'coil100', 'stanford_dogs', 'rock_paper_scissors', 'tf_flowers', 
                'cycle_gan_ukiyoe2photo', 'cassava', 'fashion_mnist', 'emnist_mnist', 'cmaterdb_telugu', 'malaria', 'eurosat_rgb', 
                'emnist_balanced', 'cars196', 'cycle_gan_iphone2dslr_flower', 'cycle_gan_summer2winter_yosemite', 'cats_vs_dogs']

train_datasets = ['cifar100', 'cycle_gan_vangogh2photo', 'uc_merced', 'cifar10', 'cmaterdb_devanagari', 
                  'cmaterdb_bangla', 'mnist', 'horses_or_humans', 'kmnist', 'cycle_gan_horse2zebra', 'cycle_gan_facades', 
                  'cycle_gan_apple2orange', 'cycle_gan_maps', 'imagenette', 'emnist_byclass', 
                  'svhn_cropped', 'coil100', 'stanford_dogs', 'rock_paper_scissors', 'tf_flowers', 
                  'cycle_gan_ukiyoe2photo', 'cassava', 'fashion_mnist', 'emnist_mnist', 'cmaterdb_telugu', 'malaria', 'eurosat_rgb',
                  'emnist_balanced', 'cars196', 'cycle_gan_iphone2dslr_flower', 'cycle_gan_summer2winter_yosemite', 'cats_vs_dogs']

val_datasets = ['imagenet_resized_32x32', 'omniglot', 'colorectal_histology']

nascomp_dev = ['devel_dataset_0', 'devel_dataset_1', 'devel_dataset_2']

# TO DO
natural = []
artificial = []
medical = []
satellite = []

GROUPS = {
          'all': all_datasets, 
          'training': train_datasets, 
          'validation': val_datasets,
          'natural': natural,
          'artificial': artificial,
          'medical': medical,
          'satellite' : satellite,
          'nascomp_dev': nascomp_dev
         }

if __name__ == '__main__':

    for group, elements in GROUPS.items():
      print('Dataset group {} contains {} dataset(s). These are ->'.format(group, len(elements)))
      print('\n'.join(elements))
      print('='*60)
