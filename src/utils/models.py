import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

def baseline_plm(dropout = 0.2):
    model = tf.keras.Sequential(
    [
    tf.keras.layers.Dense(512, activation = 'relu', input_shape = (1024,)),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
]
    )
    return model
def baseline_oh(dropout = 0.2):
    model = tf.keras.Sequential(
    [
    tf.keras.Input(shape=(2000,22)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
]
    )
    return model
def mult_DNN(tensorshape, outneuron = 14, dropout = 0.2):
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation = 'relu', input_shape = (tensorshape,)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(outneuron, activation='softmax')
    ])
        return model
def run_model_fit(model, x, y, batch, lr, epoch):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=epoch, batch_size=batch, verbose = 0)
    return model

def opt_youdenJ(True_vals, Pred_vals):
    fpr, tpr, thresholds = roc_curve(True_vals, Pred_vals)
    optim_idx = np.argmax(tpr - fpr)
    optim_threshold = thresholds[optim_idx]
    return optim_threshold, fpr, tpr

def get_performance(model, test_x, test_y):
    y_pred = model.predict(test_x, verbose = 0)
    auc = roc_auc_score(test_y, y_pred)
    
    truelist = test_y.tolist()
    predlist = y_pred.tolist()
    YoudenJ, _, _ = opt_youdenJ(truelist, predlist)
    
    return auc, YoudenJ

def eval_mult(model, test_x, test_y):
    pred = model.predict(test_x, verbose = 0)
    pred_idx = [np.argmax(val) for val in pred]
    
    report = classification_report(test_y, pred_idx, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df

def build_mult(train_x, train_y, val_x, val_y, BATCHSIZE, LEARNRATE, PATIENCE):
    model = mult_DNN(train_x.shape[1])
    opt = tf.keras.optimizers.Adam(learning_rate= LEARNRATE)
    model.compile(
        optimizer = opt,
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy', 'AUC']
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, start_from_epoch=5)
    model.fit(train_x, train_y, epochs = 50, validation_data=(val_x, val_y), callbacks= [callback], batch_size = BATCHSIZE)
    
    return model