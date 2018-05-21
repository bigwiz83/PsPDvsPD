from keras.layers import Dense, LSTM, Merge, Convolution2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential, model_from_json
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler

from scipy import interp
import pandas as pd
from pandas import DataFrame

import tensorflow as tf
import numpy as np
import dicom

import os.path
from matplotlib import pyplot as plt
from datetime import datetime

import itertools
import random


### For tuning for stochastic model
# seed=random.randint(0, 2**31 - 1)
### For reproductivity in 10-fold
# seed= 205486557
### For Benchmark
seed=7777


print(seed)
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)

### The number of input images
timesteps = 9

### FOV and pixel size
FOV_row = 200
FOV_col = 200
frame_row = 256
frame_col = 256
channels = 1

### The number of clinical features
nb_clinic = 7

### The number of class: PsPD versus PD
nb_class = 1


def clinic_model():
    clinic = Sequential()
    clinic.add(Flatten(input_shape=[nb_clinic, 1]))
    clinic.add(Dense(4, activation='relu'))
    clinic.add(Dense(4, activation='relu'))

    return clinic

def mri_model():
    mri1 = Sequential()
    mri1.add(TimeDistributed(BatchNormalization(), input_shape=[None, frame_row, frame_col, channels]))
    mri1.add(TimeDistributed(Convolution2D(64, (2, 2), padding='same', activation='relu')))
    mri1.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    mri1.add(TimeDistributed(BatchNormalization()))
    mri1.add(TimeDistributed(Convolution2D(128, (2, 2), padding='same', activation='relu')))
    mri1.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    mri1.add(TimeDistributed(BatchNormalization()))
    mri1.add(TimeDistributed(Convolution2D(256, (2, 2), padding='same', activation='relu')))
    mri1.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    mri1.add(TimeDistributed(Flatten()))
    mri1.add(LSTM(24, activation='sigmoid'))

    mri1.add((Dense(32, activation='relu')))
    mri1.add((Dense(32, activation='relu')))

    return mri1

def mri_model (ncell):
    mri1 = Sequential()
    mri1.add(TimeDistributed(BatchNormalization(), input_shape=[None, frame_row, frame_col, channels]))
    mri1.add(TimeDistributed(Convolution2D(64, (2, 2), padding='same', activation='relu')))
    mri1.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    mri1.add(TimeDistributed(BatchNormalization()))
    mri1.add(TimeDistributed(Convolution2D(128, (2, 2), padding='same', activation='relu')))
    mri1.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    mri1.add(TimeDistributed(BatchNormalization()))
    mri1.add(TimeDistributed(Convolution2D(256, (2, 2), padding='same', activation='relu')))
    mri1.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    mri1.add(TimeDistributed(Flatten()))
    mri1.add(LSTM(ncell, activation='sigmoid'))

    mri1.add((Dense(32, activation='relu')))
    mri1.add((Dense(32, activation='relu')))

    return mri1



def makemodel (ncell):
    decoder = Sequential()
    decoder.add(Merge([mri_model(ncell), clinic_model()], mode='concat'))
    decoder.add(Dense(1, activation='sigmoid'))

    decoder.compile(loss='binary_crossentropy',
                     optimizer='sgd',
                     metrics=['accuracy'])

    return decoder

def makemodel_wo_clinic (ncell):
    decoder=mri_model(ncell)
    decoder.add(Dense(1, activation='sigmoid'))

    decoder.compile(loss='binary_crossentropy',
                     optimizer='sgd',
                     metrics=['accuracy'])


    return decoder


def determine_epoch (x_mri1, x_clinic, y):
## Optimal epoch was determined to 25 with 20%.
## Epoch is tightly associated with learning rate

    train_param = DataFrame()
    val_param = DataFrame()

    for i in range(5):
        decoder = makemodel()
        history=decoder.fit([x_mri1, x_clinic], y, batch_size=7, epochs=100,
                               validation_split=0.2, shuffle=False)
        train_param[str(i)] = history.history['loss']
        val_param[str(i)] = history.history['val_loss']

    plt.plot(train_param, color='blue', label='train')
    plt.plot(val_param, color='orange',label='validation')
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def determine_batch (x_mri1, x_clinic, y):
    params = [9,8,7,6]  # with ncell = 24 -> Determine 8
    ncell = 24

    k = 5
    scores = DataFrame()
    i = 0

    cv = StratifiedKFold(n_splits=k, shuffle=False, random_state=seed)
    for value in params:
        aucs = list()

        for train, test in cv.split(x_clinic, y):
            decoder = makemodel(ncell)
            decoder.fit([x_mri1[train], x_clinic[train]], y[train], batch_size=value, epochs=25, verbose=2)
            probas_ = decoder.predict([x_mri1[test], x_clinic[test]], batch_size=value)
            # Compute ROC curve and area
            # the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            i+=1
            print('>%d/%d batch=%f auc=%f', (i + 1, k, value, roc_auc))
            K.clear_session()

        scores[str(value)] = aucs

    print(scores.describe())
    scores.boxplot()
    plt.show()
    scores.to_csv('result for batch.csv')
    (scores.describe()).to_csv('result_sum for batch.csv')

def determine_ncell (x_mri1, x_clinic, y):

# Optimal ncell size usign 5-fold validation in testing set, comparing AUC
    params = [24, 22, 20, 18] # with batch=8 --> determinined to 24

    k=5
    scores = DataFrame()
    i = 0

    cv = StratifiedKFold(n_splits=k, shuffle=False, random_state=seed)
    for value in params:
        aucs = list()

        for train, test in cv.split(x_clinic, y):
            decoder = makemodel(value)
            decoder.fit([x_mri1[train], x_clinic[train]], y[train], batch_size=6, epochs=30, verbose=2)
            probas_ = decoder.predict([x_mri1[test], x_clinic[test]], batch_size=6)
            fpr, tpr, thresholds = roc_curve(y[test], probas_)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            print('%d iteration, No of LSTM cell=%f, ROC=%f' % (i+1, value, roc_auc))
            i = i+1
            K.clear_session()

        scores[str(value)] = aucs

    print(scores.describe())
    scores.boxplot()
    plt.show()

    scores.to_csv('result for ncell.csv')
    (scores.describe()).to_csv('result_sum for ncell.csv')


def run_train():

    TRAIN_FOLDER = '<To be modified>'
    x_clinic, y = load_csv(TRAIN_FOLDER)
    # x_mri1 = load_dicom(TRAIN_FOLDER, test=False)
    x_mri1=load_npz('Training/input_data_train.npz')


    ######## Parameter Tuning ###########
    # determine_epoch(x_mri1, x_clinic, y)
    # determine_batch(x_mri1, x_clinic, y)
    # determine_ncell(x_mri1, x_clinic, y)


    ##########################################################################
    ## 5-fold internal validation with 1 iteration in the trainingining set  #
    ##########################################################################

    scores_roc = DataFrame()
    meanROC = list()

    scores_precision = DataFrame()
    meanPre = list()


    k = 10
    iter = 1
    cv = StratifiedKFold(n_splits=k, shuffle=False, random_state=seed)

    for i in range(iter):
        aucs_roc, aucs_precall= cal_auc(cv, x_clinic, x_mri1, y)
        scores_roc[str(i)] = aucs_roc
        meanROC.append(np.mean(aucs_roc))

        scores_precision[str(i)] = aucs_precall
        meanPre.append(np.mean(aucs_precall))


    if i>1:

        print(scores_roc.describe())
        scores_roc.boxplot()
        plt.show()
        plt.clf()

        print(scores_precision.describe())
        scores_precision.boxplot()
        plt.show()

        # confidence intervals
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(meanROC, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(meanROC, p))
        print('AUC: Grand Mean = %.1f, %.1f confidence interval %.1f%% and %.1f%%' % (np.mean(meanROC), alpha * 100, lower * 100, upper * 100))

        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(meanPre, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(meanPre, p))
        print('AUPRC: Grand Mean = %.1f, %.1f confidence interval %.1f%% and %.1f%%' % (np.mean(meanPre), alpha * 100, lower * 100, upper * 100))

        scores_roc.to_csv('result_roc.csv')
        (scores_roc.describe()).to_csv('result_sum_roc.csv')

        scores_precision.to_csv('result_pre.csv')
        (scores_precision.describe()).to_csv('result_sum_pre.csv')



def finalize_model():

    TRAIN_FOLDER = 'C:/Users/JBS/Desktop/PsPDtrain_1mm(SNUH)'
    x_clinic, y = load_csv(TRAIN_FOLDER)
    # x_mri1 = load_dicom(TRAIN_FOLDER, test=False)
    x_mri1=load_npz('input_data_train.npz')


    NCELL=24
    EPOCH=25
    BATCH=8



    decoder=makemodel(NCELL)

    architecture = decoder.to_json()
    with open('Structure.json', 'wt') as json_file:
        json_file.write(architecture)

    decoder.fit([x_mri1, x_clinic], y, batch_size=BATCH, epochs=EPOCH, shuffle=False)
    decoder.save_weights('final.hdf5')

def finalize_model_wo_clinic():

    TRAIN_FOLDER = '<to be modified>'
    x_clinic, y = load_csv(TRAIN_FOLDER)
    # x_mri1 = load_dicom(TRAIN_FOLDER, test=False)
    x_mri1=load_npz('Training/input_data_train.npz')

    NCELL=24
    EPOCH=25
    BATCH=8

    decoder=makemodel_wo_clinic((NCELL))

    architecture = decoder.to_json()
    with open('Finalized Models/Structure_wo.json', 'wt') as json_file:
        json_file.write(architecture)

    decoder.fit(x_mri1, y, batch_size=BATCH, epochs=EPOCH, shuffle=False)
    decoder.save_weights('Finalized Models/final_wo.hdf5')



def run_test():

    TEST_FOLDER = '<to be modified>'
    x_clinic, y = load_csv(TEST_FOLDER)
    # x_mri1 = load_dicom(TEST_FOLDER)
    x_mri1 = load_npz('Testing/input_data_test.npz')

    json_file = open('Finalized Models/Structure.json', 'rt')
    architecture = json_file.read()
    json_file.close()

    models = model_from_json(architecture)

    models.load_weights('Finalized Models/final.hdf5')
    show_auc(models, x_mri1, x_clinic, y)

def run_test_wo_clinic():

    TEST_FOLDER = 'C:/Users/JBS/Desktop/PsPDtest_1mm(SNUBH)'
    x_clinic, y = load_csv(TEST_FOLDER)
    # x_mri1 = load_dicom(TEST_FOLDER)
    x_mri1 = load_npz('Testing/input_data_test.npz')


    json_file = open('Finalized Models/Structure_wo.json', 'rt')
    architecture = json_file.read()
    json_file.close()

    models = model_from_json(architecture)

    models.load_weights('Finalized Models/final_wo.hdf5')
    show_auc_wo_clinic(models, x_mri1, y)


def cal_auc(cv, x_clinic, x_mri1, y):
    tprs = []
    aucs_roc = []
    aucs_precall = []
    ytests =[]
    probas = []
    precision = dict()
    recall = dict()
    average_precision = dict()


    mean_fpr = np.linspace(0, 1, 100)
    i = 0

    for train, test in cv.split(x_clinic, y):
        classifier = makemodel(24)
        classifier.fit([x_mri1[train], x_clinic[train]], y[train], batch_size=8, epochs=25, verbose=2)

        probas_ = classifier.predict([x_mri1[test], x_clinic[test]], batch_size=8)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs_roc.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC of fold %d (AUC = %0.2f)' % (i, roc_auc))

        # Compute precision-recall curve
        precision[i], recall[i], _ = precision_recall_curve(y[test], probas_)
        average_precision[i] = average_precision_score(y[test], probas_)
        ytests.append(y[test])
        probas.append(probas_)

        i += 1
        K.clear_session()


    ################################################
    ###########   Plotting ROC curve  ##############
    ################################################
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_roc)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.show()
    plt.savefig('Mean ROC (AUC = %0.2f -- %0.2f).pdf' % (mean_auc, std_auc), format='pdf')
    plt.clf()

    ################################################
    ##### Plotting Precision-Recall curve ##########
    ################################################

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(np.concatenate(ytests),np.concatenate(probas))
    average_precision["micro"] = average_precision_score(y[test], probas_, average="micro")

    plt.clf()
    plt.plot(recall["micro"], precision["micro"],
             label='micro-average Precision-recall curve (AUPRC = {0:0.2f})'
                   ''.format(average_precision["micro"]))

    for j in range(i):
        plt.plot(recall[j], precision[j],
                 label='Precision-recall curve of fold {0} (AUPRC = {1:0.2f})'
                       ''.format(j, average_precision[j]))
        aucs_precall.append(average_precision[j])


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    currenttime = datetime.now().strftime("%H%M%S")
    plt.savefig('Precision-Recall Curve (AUPRC = %0.2f)_' % (average_precision["micro"]) + currenttime +'.pdf' , format='pdf')
    plt.show()
    plt.clf()

    return aucs_roc, aucs_precall

def show_auc(classifier, x_mri1, x_clinic, y):

    probas_ = classifier.predict([x_mri1, x_clinic], batch_size=8)

    # Compute ROC curve
    fpr, tpr, threshold = roc_curve(y, probas_)
    tpr[0]=0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC Curve (AUC = %0.2f)' % (roc_auc), alpha=1, color='b')

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y, probas_)
    average_precision = average_precision_score(y, probas_)

    ################################################
    ###########   Plotting ROC curve  ##############
    ################################################
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, alpha=0.2, color='r',label='Luck')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")

    plt.savefig('Final ROC (AUC = %0.2f).pdf' % (roc_auc), format='pdf')
    plt.show()
    plt.clf()

    ################################################
    ##### Plotting Precision-Recall curve ##########
    ################################################

    # Compute micro-average ROC curve and ROC area

    plt.plot(recall, precision,label='Precision-Recall Curve (AUPRC = {0:0.2f})'''.format(average_precision))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig('Final Precision-Recall (AUPRC = %0.2f).pdf' % (average_precision), format='pdf')
    plt.show()
    plt.clf()


    ################################################
    ##### Confusion Matrix #########################
    ################################################


    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point

    i = np.arange(len(tpr))  # index for df
    roc = pd.DataFrame(
        {'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1 - fpr, index=i),
         'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc.ix[(roc.tf - 0).abs().argsort()[:1]]

    #### Confusion Matrix
    roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]
    optimal_cutoff = list(roc_t['threshold'])
    print (optimal_cutoff)
    report = classification_report(y, probas_ >= optimal_cutoff)
    print (report)

    cm = confusion_matrix(y, probas_ >= optimal_cutoff)
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    fmt = '.2f'
    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

def show_auc_wo_clinic (classifier, x_mri1, y):

    probas_ = classifier.predict([x_mri1], batch_size=8)

    # Compute ROC curve
    fpr, tpr, threshold = roc_curve(y, probas_)
    tpr[0]=0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC Curve (AUC = %0.2f)' % (roc_auc), alpha=1, color='orange')

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y, probas_)
    average_precision = average_precision_score(y, probas_)

    ################################################
    ###########   Plotting ROC curve  ##############
    ################################################
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, alpha=0.2, color='r',label='Luck')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")

    plt.savefig('Final ROC wo clinic (AUC = %0.2f).pdf' % (roc_auc), format='pdf')
    plt.show()
    plt.clf()

    ################################################
    ##### Plotting Precision-Recall curve ##########
    ################################################

    # Compute micro-average ROC curve and ROC area

    plt.plot(recall, precision,label='Precision-Recall Curve (AUC = {0:0.2f})'''.format(average_precision), color='orange')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig('Final Precision-Recall wo clinic(AUC = %0.2f).pdf' % (average_precision), format='pdf')
    plt.show()
    plt.clf()


    ################################################
    ##### Confusion Matrix #########################
    ################################################


    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point

    i = np.arange(len(tpr))  # index for df
    roc = pd.DataFrame(
        {'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1 - fpr, index=i),
         'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc.ix[(roc.tf - 0).abs().argsort()[:1]]

    #### Confusion Matrix
    roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]
    optimal_cutoff = list(roc_t['threshold'])
    print (optimal_cutoff)
    report = classification_report(y, probas_ >= optimal_cutoff)
    print (report)

    cm = confusion_matrix(y, probas_ >= optimal_cutoff)
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    plt.matshow(cm, cmap=plt.cm.Oranges)
    plt.colorbar()
    fmt = '.2f'
    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

def plot_history (history):

    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


def load_csv(path):
    pathfile = path+'/sample.csv'
    dataframe = pd.read_csv(pathfile)
    dataset = dataframe.values


    X = np.array (dataset[:, 1:nb_clinic+1].astype(float))
    Y = np.array(dataset[:, nb_clinic+1])


    scaler = MinMaxScaler (feature_range=(0,1))
    scalerX = scaler.fit(X)
    normalizedX = scalerX.transform(X)

    X = normalizedX.reshape(normalizedX.shape[0], normalizedX.shape[1], 1)



    return X,Y

def load_dicom(path, test=False):


    mark = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008']
    n = len(os.listdir(path)) - 1
    j = 0

    stack = np.zeros((timesteps, frame_row, frame_col))
    data = np.zeros((n, timesteps, frame_row, frame_col))

    with tf.Session() as sess:
        for ptnum in os.listdir(path):
            if ptnum == 'sample.csv':
                continue
            else:
                for i in mark:
                    dcm_struct=dicom.read_file(path + '/' + ptnum + '/' + ptnum + '_' + i + '.dcm')
                    pixel_size=dcm_struct.PixelSpacing[0]
                    img = dcm_struct.pixel_array
                    plt.imshow(img, cmap=plt.cm.gray)
                    img = img.reshape(img.shape[0],img.shape[1],1)
                    img = img.astype(int)
                    print(i,j)
                    img = tf.image.resize_image_with_crop_or_pad(img, int(round(FOV_row/pixel_size)), int(round(FOV_col/pixel_size)))
                    img = tf.image.resize_images(img, size=[frame_row, frame_col], method=tf.image.ResizeMethod.BICUBIC)
                    img = tf.image.per_image_standardization(img)

                    img_array = sess.run(img)
                    img_array = img_array.reshape(img.shape[0], img.shape[1])
                    plt.imshow(img_array, cmap=plt.cm.gray)
                    stack[int(i), :, :] = img_array

            data [j,:,:,:] = stack
            j = j+1


    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3], channels)

    if test==True:
        np.savez_compressed('input_data_test.npz', data)
    else:
        np.savez_compressed('input_data_train.npz', data)

    # show_slice(data[0][4])

    return data

def load_npz(file):
    data=np.load(file)
    data=data['arr_0']
    return data

def show_slice(arr, value_range=None):
    if len(list(arr.shape)) > 2:
        arr2 = arr.copy()
        arr2 = np.reshape(arr, (arr.shape[0], arr.shape[1]))
    else:
        arr2 = arr

    dpi = 80
    margin = 0.05  # (5% of the width/height of the figure...)
    xpixels, ypixels = arr2.shape[0], arr2.shape[1]

    figsize = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])


    if value_range is None:
        plt.imshow(arr2, cmap=plt.cm.gray)
    else:
         plt.imshow(arr2, vmin=value_range[0], vmax=1, cmap=plt.cm.gray, interpolation='none')

    plt.show()


if __name__ == '__main__':
     # run_train()
     # finalize_model()
     run_test()

     # finalize_model_wo_clinic()
     run_test_wo_clinic()


