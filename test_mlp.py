import pandas as pd
import numpy as np
from keras.losses import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


from keras.callbacks import ModelCheckpoint
from keras import optimizers, metrics
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from dataset import Datasets
from report import ReportGenerator
from keras import backend as K

# Here it is an array of optimizers with various params to test diff approazhes of learning
optimizes = [
    # optimizers.SGD(lr=0.01, decay=4e-7, momentum=0.9, nesterov=True),
    optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
    # optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0),
    # optimizers.RMSprop(lr=0.001, rho=0.8, epsilon=1e-08, decay=0.0),
    # optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),
    # optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
    # optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
    # optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
    # optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
]

# Here it is Fields of dataset
fields = [
'weekend',
' coverphoto',
' event',
' link',
' photo',
' question',
' status',
' swf',
' video',
'coverphoto_reactions',
'coverphoto_comments',
'coverphoto_shares',
'event_reactions',
'event_comments',
'event_shares',
'link_reactions',
'link_comments',
'link_shares',
'photo_reactions',
'photo_comments',
'photo_shares',
'question_reactions',
'question_comments',
'question_shares',
'status_reactions',
'status_comments',
'status_shares',
'swf_reactions',
'swf_comments',
'swf_shares',
'video_reactions',
'video_comments',
'video_shares'
]
_FILENAMES = [
    'Dodge_result.csv',
    'Ferrari_result.csv',
    'Hyundai_result.csv',
    'Kia_result.csv',
    'Mercedes-Benz_result.csv',
    'Mini_result.csv',
    'Peugeot_result.csv',
    'Toyota_result.csv',
    'Volkswagen_result.csv'
]
# Object that returns splitted datasets by brand name
ds = Datasets()
brands_datasets = ds.getByBrands()

for brand_file,res in brands_datasets.items():
    brand = brand_file.split('_')[0]
    y = res[[' change']].as_matrix()
    x = res[fields].as_matrix()

    # It's normalization technique that scales all input components together
    # scaler_x = PCA(n_components=int(len(fields))).fit(x)
    # std_x = scaler_x.transform(x)

    scaler_x = StandardScaler().fit(x)
    std_x = scaler_x.transform(x)

    # scaler_x = MinMaxScaler().fit(x)
    # std_x = scaler_x.transform(x)

    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    scaler_y = MinMaxScaler().fit(y)
    std_y = scaler_y.transform(y)
    # scaler_y = StandardScaler().fit(y)
    # std_y = scaler_y.transform(y)

    # Custom loss\cost function that minimizes the difference of origin absolute values
    def c_abs_metric(y_true, y_pred):
        scale_ = K.variable(value=scaler_y.scale_)
        min_ = K.variable(value=scaler_y.min_)
        orig_true_y = (y_true-min_)/scale_
        orig_pred_y = (y_pred-min_)/scale_
        diff = K.abs(orig_pred_y - orig_true_y)
        max = K.maximum(K.maximum(K.abs(orig_pred_y),K.abs(orig_true_y)),1)
        # max = K.maximum(K.abs(orig_true_y),K.abs(orig_pred_y))
        return K.mean(diff/max, axis=-1)


    # x_other,x_test,y_other,y_test = train_test_split(std_x,std_y,test_size=0.33,random_state=42)
    #Splitting the input data into  test (7 last records)and others for training
    x_other = std_x[:-7,:]
    y_other = std_y[:-7,:]
    x_test = std_x[-7:,:]
    y_test = std_y[-7:,:]
    #Splitting the training data to 2 datasets namely validation and training
    x_train,x_valid,y_train,y_valid = train_test_split(x_other,y_other,test_size=0.33,random_state=42)

    loss = 'mean_absolute_error'
    # opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    models = {}
    for opt in optimizes:
        model = Sequential()
        model.add(Dense(len(fields), input_dim=std_x.shape[1], kernel_initializer='uniform', activation='relu'))
        model.add(Dense(int(std_x.shape[1]/2), kernel_initializer='uniform', activation='relu'))
        model.add(Dense(int(std_x.shape[1]/4), kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        try:
            model.compile(
                optimizer=opt,
                # loss=c_abs_metric,
                # metrics=[c_abs_metric,mean_absolute_error]
                loss=loss,
                metrics=[mean_absolute_error]

            )
            early_stopping_monitor = EarlyStopping(patience=1)
            # checkpoint = ModelCheckpoint('./checkpoints/mlp.mdl',save_best_only=False)
            model.fit(x_train,
                      y_train,
                      batch_size=1,
                      epochs=25,
                      verbose=1,
                      validation_data=(x_valid, y_valid),
                      callbacks=[
                          early_stopping_monitor,
                          # checkpoint
                        ]
                      )
            score = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
            print(brand)
            print(loss)
            print(opt)
            print(score)
            models[brand] = model
        except Exception as e:
            print(e)
    # It's a simple report generator
    genRep = ReportGenerator('./output/'+brand+'/', y_test, x_test, scaler_y)
    genRep.generate(models)