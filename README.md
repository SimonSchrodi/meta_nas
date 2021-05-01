# meta_nas

## Overview
The [google doc](https://docs.google.com/document/d/1vNX8wBWLw41LELCP5N8NP-tKJc8Ky6jDC0jSqA9ipH0/edit) contains all information regarding the project.

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

## Submission record
[The google doc](https://docs.google.com/document/d/1rntplrNAvPcGvdIzz1-0dxHnkZvVoUhddyxZYJ-q_vc/edit) keeps track of our submissions. For development purposes of the meta learner, we can additionally track performance of individual models to improve it offline.