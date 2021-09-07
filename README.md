# M3I Graph Pooling

==========

The source code for submission-654 of AAAI 2022: Hierarchical Unsupervised Graph Pooling via Multi-Granularity Mutual Information Maximization

==========

Dependencies

-Python3.x

-torch

-tqdm

-networkx

==========

type ./run_MIP.sh DATANAME FOLD

to run MIP on dataset = DATANAME using fold number = FOLD (1-10, corresponds to which fold to use as test data in the cross-validation experiments).

If you set FOLD = 0, e.g., typing "./run_MIP.sh MUTAG 0", then it will run 10-fold cross validation on MUTAG and report the average accuracy.

==========

check "run_MIP.sh" for more options
