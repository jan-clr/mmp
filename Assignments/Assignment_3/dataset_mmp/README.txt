Directory structure
--------------------------------
The dataset is partitioned into a training and test set. These sets can be
found in the directories "train" and "test", respectively. The training set can
be used in any way the user sees fit. In particular, if a separate validation
set is needed, it can be obtained as a subset of the training set.
The test set is to be used for evaluation purposes only.


Groundtruth data format
--------------------------------
For each image there exists a text file with the extension .gt_data.txt which
contains the object annotations. 

Each line in this file corresponds to an object instance and is to be
interpreted in the following way:

<x1> <y1> <x2> <y2> <class_id> <dummy_value> <mask> <difficult> <truncated>

Since this dataset only contains annotations for a single object class,
the only class_id is always 0.

All other information can and should be disregarded for the purpose of this project.
