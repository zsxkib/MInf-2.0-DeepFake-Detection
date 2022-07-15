# Appendices

### Notice

This chapter does not intend to infringe on the 40-page limit.
Everything included from here onwards is supplementary and not vital to
the understanding of this project. To view the table of all results
ordered by test AUC, see .

### Code

We **strongly encourage** the reader of this paper to take a look at our
[GitHub](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection),
specifically
[`./Collate_Results.ipynb`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/blob/main/Collate_Results.ipynb)
which contains the bulk of the code used for analysing.

This GitHub contains all code used for experimentation. However, we
exclude the datasets and saved weights for each trained model as they
are too large in size to fit to onto our repository.  

* [`./_BASELINE_TESTS/`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/tree/main/_BASELINE_TESTS)
contains code for all baselines experiments.

  * [`./_BASELINE_TESTS/Results/`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/tree/main/_BASELINE_TESTS/Results)
contains individual AUCs, F1s, y\_true, y\_pred, etc. as text files for
all baselines experiments.

* [`./_DATASETS/`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/tree/main/_DATASETS)
would usually contain all pre-processed datasets but this was omitted
due to GitHub’s repository file limit.

  * [`./_DATASETS/FaceForensicspp/pipeline.py`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/blob/main/_DATASETS/FaceForensicspp/pipeline.py)
contains our pre-processing algorithms as a set of helper functions
within one python file. Please see `get_stable_faces()` function to
see the $T_{70}$ protocol described in and .

  * [`./FaceForensicspp/preprocess_ffpp.ipynb`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/blob/main/_DATASETS/FaceForensicspp/preprocess_ffpp.ipynb)
contains the code to pre-process FF++ into AF and RF datsets. The same
code was used to pre-process CDFv2.

* [`./_PLOTS/`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/tree/main/_PLOTS)
contains all plots generated for our analysis.

* [`./_TRAINING/ViT/`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/tree/main/_TRAINING/ViT)
contains all code to train and test our ViT models.
  * [`./_TRAINING/ViT/Results/`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/tree/main/_TRAINING/ViT/Results)
contains individual AUCs, F1s, y\_true, y\_pred, etc. as text files for
all ViT experiments.

* [`./_TRAINING/OC-FakeDect-Implementation/`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/tree/main/_TRAINING/OC-FakeDect-Implementation)
contains all code to train and test our OC-FakeDect1 models.

  * [`./_TRAINING/OC-FakeDect-Implementation/Results/`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/tree/main/_TRAINING/OC-FakeDect-Implementation/Results)
contains individual AUCs, F1s, y\_true, y\_pred, etc. as text files for
all OC-FakeDect1 experiments.

* [`./_WEIGHTS/`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/tree/main/_WEIGHTS/)
would usually contain all saved model weights but this was omitted due
to GitHub’s repository file limit.

* [`./env.yaml`](https://github.com/Sakib56/MInf-2.0-DeepFake-Detection/blob/main/env.yaml)
is the environment used to produce all of these results, in contains
libraries used and their source. All experiments were either done using
a Tesla P100 via Google Colab Pro or an RTX 3070 at home, both utilising
16GB RAM.

If you would like to access our saved model weights, pre-proccessed
datsets, or any other part of this project. Please email at
<zsakib.ahamed@gmail.com> and we will send you the relevant files.
Please note that all these files total 63GB.
