# Emoji Recommendation
This project trains models to recommend suitable emojis for sentences
such as microblogs.

We implemented and compared three algorithms:
- Multinomial Navie Bayes
- OvO Linear SVM
- CNN

We introduce below how to train models by those algorithms.

## Dependencies
- numpy (1.15.4)
- scipy (1.1.0)
- pandas (0.23.4)
- jieba (0.39)
- scikit-learn (0.20.1)
- pytorch (1.0.1)

Since my environment has other useless packages and I am lazy to create
a new enviroment, find the minimum dependency, and freeze the dependency
to `requirment.txt`, you have to install the packages one by one with the
version provided above. If you encounter some errors, you may have to fix
it by yourself. Besides, the newest minor version for each package with
major version fixed is a good try.

## Dataset
Our raw data include train and test datasets. You should create a `dataset`
directory under the root of this project and unzip `raw/dataset.zip` into it.

You should also download pretrained word embeddings from
`https://github.com/Embedding/Chinese-Word-Vectors`
and unzip it into the `dataset` directory if you want to train a model by cnn.

## Methods

### Preprocess
Before running any algorithms to train a model, you have to transform
the raw data to what can be accepted by the algorithm. This can be done by

    python preprocess.py ALGS

where ALGS is the algorithm you want to use and its value can be one of
the follows:
- nb
- lsvm
- cnn

### Multinomial Naive Bayes
You can see the validation results of multinomial naive bayes
based on the train dataset by

    python nb.py val

Learn more usages by `python nb.py -h`. Here is an overview:

    python nb.py [-h] [-a ALPHA] [-f FOLD] [-r RESULT] {val,cv,pred}

### OvO Linear SVM
You can see the validation results of linear svm
based on the train dataset by

    python lsvm.py val

Learn more usage by `python lsvm.py -h`. Here is an overview:

    python lsvm.py [-h] [-c C] [-f FOLD] [-r RESULT] [-v] {val,cv,pred}

### CNN
You can train the model with default parameters by

    python cnn.py train --name NAME

and the model will be save as `model/NAME.pt`.
After training, you can use

    python cnn.py test --name NAME --result RESULT

to predict results for test dataset by model `model/NAME.pt` and the results
will be output to `result/RESULT`.

Learn more usage by `python lsvm.py -h`. Here is an overview:

    python cnn.py [-h] [-n NAME] [-d DROPOUT] [-k KERNEL_NUM]
              [-s KERNEL_SIZES [KERNEL_SIZES ...]] [-e EPOCH] [-b BATCH]
              [-l LR] [-r RESULT]
              {train,test}

The default values for the arguments are listed below:
- NAME='model'
- DROPOUT=0.5
- KERNEL_NUM=200
- KERNEL_SIZES=[2, 3, 4, 5]
- EPOCH=5
- BATCH=100
- LR=1e-4
- RESULT=cnn_result.csv

### Ensemble

We also provide a way to ensemble results predicted from different models.
This can be done by

    python ensemble.py -b BASELINE [-o OUTPUT] inputs [inputs ...]

where `inputs` are results from different models and `baseline` is used to
resolve conflicts, which can be one of the `inputs` files.
