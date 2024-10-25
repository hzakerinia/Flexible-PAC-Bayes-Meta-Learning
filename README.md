# More Flexible PAC-Bayesian Meta-Learning by Learning Learning Algorithms

 This is the implementation of the paper
  ["More Flexible PAC-Bayesian Meta-Learning by Learning Learning Algorithms"](https://proceedings.mlr.press/v235/zakerinia24a.html) , ICML 2024.

## Prerequisites

- Python 3.5+
- PyTorch 1.0+ with CUDA
- NumPy and Matplotlib


## Data
The data sets are downloaded automatically. Specify the main data path in the file 'Data_Path.py'

## Reproducing experiments in the paper:

Run these scripts to reproduce the results in the paper:

`python PriorMetaLearning/run_MPB_PermuteLabels.py`

`python PriorMetaLearning/run_MPB_ShuffledPixels.py`


Our baseline code is in this repository: https://github.com/ron-amit/meta-learning-adjusting-priors2

## Cite the paper
```
@inproceedings{zakerinia2024flexible,
  title = {More Flexible {PAC-Bayesian} Meta-Learning by Learning Learning Algorithms},
  author = {Zakerinia, Hossein and Behjati, Amin and Lampert, Christoph H.},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2024},
}
```
