# meta_nas

## Contents
 - [`starter_kit`](starter_kit) contains all of the original files in the starter kit, including all of the code used to evaluate submissions.
 - [`starter_kit/dev_submission`](starter_kit/dev_submission) is the folder where we dev our ideas.
 - [`starter_kit/unedited_sample_submission`](starter_kit/unedited_sample_submission) contains the original sample submission, but the metadata sets `full_training: true`.

## Testing our code locally.
As described in [`starter_kit/README.md`](starter_kit/README.md), it is easy for us to test our code locally.
Download the [public data](https://competitions.codalab.org/competitions/29853#participate-get_starting_kit) and place it in `starter_kit`. Then update [this line](starter_kit/Makefile#L12) with the code you want to test (e.g., `dev_submission`, `unedited_sample_submission`, etc). Then run:
```bash
cd starter_kit
make all data=public_data id=my_local_test
```

## Running SMAC to find architecture candidates
The smac implementation is based off of [nas_benchmarks](https://github.com/automl/nas_benchmarks/blob/master/experiment_scripts/run_smac.py)
This needs `smac==0.10.0`, the latest version will throw an error!
```bash
pip install smac==0.10.0
python dev_nas/run_smac.py --dataset_path ../../public_data/devel_dataset_0
```

## Acknowledgements
 - The implementations of the (non-tailored) models come from [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models).

 - Meta-training set is in [AutoDL challenge](https://hal.archives-ouvertes.fr/hal-02957135/document) format and the dataset loader scripts comes from competition starter kit: [Competition Webpage](https://autodl.lri.fr/competitions/162#learn_the_details).

 - We have used [Task2Vec](https://arxiv.org/abs/1902.03545) embeddings as meta-features. For the implementations of the meta-feature extractor and other helper functions see: [awslabs/aws-cv-task2vec](https://github.com/awslabs/aws-cv-task2vec).