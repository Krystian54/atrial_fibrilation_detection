import os
import numpy as np
import pandas as pd
import random
from fpdf import FPDF
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import keras
from keras import layers, models
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Input, BatchNormalization, Flatten, LSTM, Bidirectional, GRU, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class AFBDatasetTF:
    def __init__(self, data_path, records, augment=False, shuffle=False, batch_size=64, aug2nr=0):
        self.data_path = data_path
        self.records = records
        self.augment = augment
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.aug2nr = aug2nr
        self.rrs = []
        self.rrs_labels = []
        self._load_data()

    def _load_data(self):
        # wczytanie wszystkich afibów
        for record in self.records:
            # jeden rekord
            data = np.load(os.path.join(self.data_path, record))
            length = len(data[:, 0])
            start = 0
            segments = []
            labels = []

            step = WIN_SIZE // 3 if self.augment else WIN_SIZE

            while (start + WIN_SIZE) <= length:
                seg = data[start:start + WIN_SIZE, 0]
                lab_all = data[start:start + WIN_SIZE, 1]

                # średnia do oceny czy to afib
                lab = [1 if np.mean(lab_all) >= 0.1 else 0]

                # wyrzucenie outlierów
                if np.any(seg > OUTLIER_LIM):
                    start += step
                    continue

                # wyrzucenie nie afibów
                if lab[0] == 0:
                    start += step
                    continue

                segments.append(seg)
                labels.append(lab_all)

                start += step

            # dodanie do listy w obrębie wszystkich rekordów
            self.rrs.extend(np.array(segments))
            self.rrs_labels.extend(np.array(labels))

        # liczba afibów
        len_of_afib = len(self.rrs_labels)

        # print("afiby: ", len(self.rrs_labels))
        a = len(self.rrs_labels)

        # print(len_of_afib)
        # print(self.rrs)
        # print(self.rrs_labels)

        # wczytanie wszystkich arytmi nie afibów
        for record in self.records:
            # jeden rekord
            data = np.load(os.path.join(self.data_path, record))
            length = len(data[:, 0])
            start = 0
            segments = []
            labels = []

            step = WIN_SIZE // 3 if self.augment else WIN_SIZE
            # step = WIN_SIZE // 3 # zwiększenie nie afiba

            while (start + WIN_SIZE) <= length:
                seg = data[start:start + WIN_SIZE, 0]
                lab_all = data[start:start + WIN_SIZE, 1]

                # inne arytmie
                lab_all2 = data[start:start + WIN_SIZE, 2]
                lab2 = [1 if np.mean(lab_all2) >= 0.1 else 0]

                # średnia do oceny czy to afib
                lab = [1 if np.mean(lab_all) >= 0.1 else 0]

                # wyrzucenie outlierów
                if np.any(seg > OUTLIER_LIM):
                    start += step
                    continue

                # wyrzucenie nie nie innych arytmi
                if lab2[0] == 0:
                    start += step
                    continue

                segments.append(seg)
                labels.append(lab_all)

                start += step

            # dodanie do listy w obrębie wszystkich rekordów
            self.rrs.extend(np.array(segments))
            self.rrs_labels.extend(np.array(labels))


        # augmentacja

        len_of_no_augment = len(self.rrs_labels)
        # b = len(self.rrs_labels) - a

        # augmentacja przez mnożenie wyznaczoną ilość razy
        # for i in range(0, 1):
        #     for j in range(len_of_afib, len_of_no_augment):
        #         random_float = np.random.uniform(1-AUG_FACTOR, 1+AUG_FACTOR)
        #         self.rrs.extend(np.array([self.rrs[j] * random_float]))
        #         self.rrs_labels.extend(np.array([self.rrs_labels[j]]))

        # print("afiby i inne: ", len(self.rrs_labels))

        b = len(self.rrs_labels) - a

        # lista nagrań
        all_records = self.records
        random.shuffle(all_records)

        # wczytanie wszystkich nie afibów i nie arytmi
        for record in all_records:
            # jeden rekord
            data = np.load(os.path.join(self.data_path, record))
            length = len(data[:, 0])
            start = 0
            segments = []
            labels = []

            step = WIN_SIZE // 3 if self.augment else WIN_SIZE
            # step = WIN_SIZE * 2 # zwiększenie kroku dla różnorodności klasy Normal (jest ich dużo)

            while (start + WIN_SIZE) <= length:
                seg = data[start:start + WIN_SIZE, 0]
                lab_all = data[start:start + WIN_SIZE, 1]

                # inne arytmie
                lab_all2 = data[start:start + WIN_SIZE, 2]
                lab2 = [1 if np.mean(lab_all2) >= 0.1 else 0]

                # średnia do oceny czy to afib
                lab = [1 if np.mean(lab_all) >= 0.1 else 0]

                # wyrzucenie outlierów
                if np.any(seg > OUTLIER_LIM):
                    start += step
                    continue

                # wyrzucenie afibów i innych arytmi
                if lab[0] == 1 or lab2[0] == 1:
                    start += step
                    continue

                self.rrs.extend(np.array([seg]))
                self.rrs_labels.extend(np.array([lab_all]))

                start += step

                if len(self.rrs_labels) >= (2*len_of_afib):
                    break

        len_of_no_augment = len(self.rrs_labels)
        c = len(self.rrs_labels) - a - b

        # print("afiby i inne i normalne: ", len(self.rrs_labels))

        print("afib: ", a / (a+b+c))
        print("nie afib: ", b / (a+b+c))
        print("normal: ", c / (a+b+c))

        # print(len_of_no_augment)
        # print(self.rrs)
        # print(self.rrs_labels)

        # augmentacja przez mnożenie wyznaczoną ilość razy
        for i in range(0, self.aug2nr):
            for j in range(0, len_of_no_augment):
                random_float = np.random.uniform(1-AUG_FACTOR, 1+AUG_FACTOR)
                self.rrs.extend(np.array([self.rrs[j] * random_float]))
                self.rrs_labels.extend(np.array([self.rrs_labels[j]]))

        # print(len(self.rrs_labels))
        # print(self.rrs)
        # print(self.rrs_labels)

    def as_dataset(self):
        # zrobienie batchy
        x = np.expand_dims(np.array(self.rrs), axis=-1)
        y = np.expand_dims(np.array(self.rrs_labels), axis=-1)
        # y = np.array(self.rrs_labels)

        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(x))

        return dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

def bce_dice_weighted_loss_wrapper(bce_w=0.5, dice_w=0.5, smooth=10e-6):
    bce_loss = keras.losses.BinaryCrossentropy()
    dice_loss = dice_coef_loss_wrapper(smooth)
    def bce_dice_weighted_loss(y_true, y_pred):
        return bce_w * bce_loss(y_true, y_pred) + dice_w * dice_loss(y_true, y_pred)
    return bce_dice_weighted_loss

def dice_coef_wrapper(smooth=10e-6):
    def dice_coef(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return dice
    return dice_coef

def dice_coef_loss_wrapper(smooth=10e-6):
    dice_coef = dice_coef_wrapper(smooth)
    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)
    return dice_coef_loss

# do modelu
BATCH_SIZE = 128
OUTLIER_LIM = 3
AUG_FACTOR = 0.1
WIN_SIZE = 30

training_files_lib = os.listdir('training_files_2')
training_files = [f for f in training_files_lib if f.endswith('.npy')]
train_dataset = AFBDatasetTF('training_files_2', training_files, augment=False, shuffle=True, batch_size=BATCH_SIZE, aug2nr=0).as_dataset()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_files_lib = os.listdir('val_files_2')
val_files = [f for f in val_files_lib if f.endswith('.npy')]
val_dataset = AFBDatasetTF('val_files_2', val_files, augment=False, shuffle=True, batch_size=BATCH_SIZE, aug2nr=0).as_dataset()
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_files_lib = os.listdir('test_files')
test_files = [f for f in test_files_lib if f.endswith('.npy')]
test_dataset = AFBDatasetTF('test_files', test_files, augment=False, shuffle=True, batch_size=BATCH_SIZE, aug2nr=0).as_dataset()
test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = Sequential()
model.add(Input(shape=(WIN_SIZE, 1)))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(512, return_sequences=True)))
model.add(Bidirectional(LSTM(512, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(learning_rate=1e-4), loss=bce_dice_weighted_loss_wrapper(), metrics=[dice_coef_wrapper()])




current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # czas do nazw

early_stopping = EarlyStopping(
    monitor='val_dice_coef',
    patience=7,
    restore_best_weights=True,
    mode='max'
)


model_checkpoint = ModelCheckpoint(
    filepath=f'models/model_{current_time}/model_epoch_{{epoch:02d}}.keras',
    save_best_only=False,
    save_weights_only=False,
    save_freq='epoch',
    verbose=1
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)

def evaluate_model_on_test_set(model, test_dataset):
    y_true = []
    y_pred = []

    for x_batch, y_batch in test_dataset:
        preds = model.predict(x_batch)
        preds_binary = (preds > 0.5).astype(int)
        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend(preds_binary.flatten())

    classification_rep = classification_report(y_true, y_pred, target_names=["Non-AF", "AF"])
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return classification_rep, conf_matrix, accuracy

(x, y, z) = evaluate_model_on_test_set(model, test_dataset)

print(f"{x}\n\n{y}\n\n{z}")

def save_plots_to_pdf(history, model_filepath, model, test_dataset):

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['dice_coef'], label='Training Accuracy')
    plt.plot(history.history['val_dice_coef'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_image_path = "training_plots.png"
    plt.savefig(plot_image_path)
    plt.close()

    model_name = os.path.basename(model_filepath)
    model_date = model_name.split('_')[1].replace('.keras', '')

    classification_rep, conf_matrix, accuracy = evaluate_model_on_test_set(model, test_dataset)

    pdf_filename = f"raports/raport_{current_time}.pdf"
    pdf = FPDF()

    pdf.add_page()
    pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=16)
    pdf.image(plot_image_path, x=10, y=10, w=190)
    pdf.ln(120)
    pdf.set_font("DejaVu", size=12)
    pdf.cell(200, 10, txt="Model Evaluation Results", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("DejaVu", size=10)
    pdf.cell(200, 10, txt=f"Accuracy: {accuracy:.4f}", ln=True, align='L')
    pdf.ln(10)
    pdf.cell(200, 10, txt="Classification Report:", ln=True, align='L')
    pdf.set_font("DejaVu", size=8)
    for line in classification_rep.splitlines():
        pdf.cell(200, 5, txt=line, ln=True, align='L')
    pdf.ln(10)
    pdf.set_font("DejaVu", size=10)
    pdf.cell(200, 10, txt="Confusion Matrix:", ln=True, align='L')
    pdf.set_font("DejaVu", size=8)
    for row in conf_matrix:
        pdf.cell(200, 5, txt=str(row), ln=True, align='L')
    pdf.add_page()
    pdf.set_font("DejaVu", size=12)
    pdf.cell(200, 10, txt="Model Summary", ln=True, align='C')
    pdf.set_font("DejaVu", size=7)
    pdf.cell(200, 10, txt="Model Summary:", ln=True, align='L')

    model_summary = StringIO()
    model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
    model_summary_str = model_summary.getvalue()
    model_summary.close()

    for line in model_summary_str.splitlines():
        pdf.multi_cell(200, 10, txt=line, align='L')

    pdf.output(pdf_filename)

    os.remove(plot_image_path)

save_plots_to_pdf(history, model_checkpoint.filepath, model, test_dataset)
