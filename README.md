# Emoji Recommendation
This project trains a model to recommend suitable emojis for
sentences such as microblogs.

We used and compared three algorithms:
- Multinomial Navie Bayes
- OvO Linear SVM
- CNN

We introduce below how to train models by those algorithms.

## Dependency
- numpy
- scipy
- pandas
- jieba
- sklearn
- pytorch

Since I haven't tested which version of these dependecies is safe,
you may have to find the suitable version. However, the newest version
before 2020 is a good try.

## Dataset
Our raw data include train and test datasets.
You should create a `dataset` directory under the root of this project
and unzip `raw/dataset.zip` into it.

## Methods

### Preprocess
Before running any algorithms to train the model,
you have to prepare the data by

    python preprocess.py ALGS

where ALGS can be one of the follows:
- nb
- lsvm
- cnn

### Multinomial Naive Bayes
You can see the validation results of multinomial naive bayes
running on the train dataset by

    python nb.py val

Learn more usages by `python nb.py -h`. Here is an overview:

    python nb.py [-h] [-a ALPHA] [-f FOLD] [-r RESULT] {val,cv,pred}

### OvO Linear SVM
You can see the validation results of linear svm
running on the train dataset by

    python lsvm.py val

Learn more usage by `python lsvm.py -h`. Here is an overview:

    python lsvm.py [-h] [-c C] [-f FOLD] [-r RESULT] [-v] {val,cv,pred}

### CNN
You can train the model with default parameters by

    python cnn.py train --name NAME

and the model will be save as `model/NAME.pt`.
After training, you can use

    python cnn.py test --name NAME --result RESULT

to predict results for test dataset by model `mode/NAME.pt` and the results
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
