import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, InputExample, glue_convert_examples_to_features, TFBertForSequenceClassification

import utils
import values


def load_data(file_name):
    return pd.read_csv(file_name)


def clean_data(data):
    return data[data['text'].map(len) < 512]


def convert_data_into_input_examples(data):
    return [InputExample(guid=None, text_a=row.text, text_b=None, label=row.label) for _, row in data.iterrows()]


def split_data(data, ratio=values.data_split_ratio):
    return train_test_split(data, test_size=ratio)


def tokenize(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    return glue_convert_examples_to_features(examples=data, tokenizer=tokenizer, max_length=values.MAX_SEQ_LENGTH,
                                             task='mrpc', label_list=values.labels_list)


def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=values.learning_rate,
                                         epsilon=values.epsilon,
                                         clipnorm=values.clipnorm)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


def plot_learning_process(learning_process):
    acc = learning_process.history['accuracy']
    val_acc = learning_process.history['val_accuracy']
    loss = learning_process.history['loss']
    val_loss = learning_process.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='training')
    plt.plot(x, val_acc, 'r', label='validation')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='training')
    plt.plot(x, val_loss, 'r', label='validation')
    plt.title('Training and validation loss')
    plt.legend()


def my_solution(bdset):
    input_ids, attention_mask, token_type_ids, label = [], [], [], []
    for in_ex in bdset:
        input_ids.append(in_ex.input_ids)
        attention_mask.append(in_ex.attention_mask)
        token_type_ids.append(in_ex.token_type_ids)
        label.append(in_ex.label)

    input_ids = np.vstack(input_ids)
    attention_mask = np.vstack(attention_mask)
    token_type_ids = np.vstack(token_type_ids)
    label = np.vstack(label)
    return [input_ids, attention_mask, token_type_ids], label


def example_to_features(input_ids, attention_masks, token_type_ids, y):
    return {"input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids}, y


if __name__ == "__main__":
    data_file_path = utils.get_data_file_name()
    _data = clean_data(load_data(data_file_path))
    print('\n\n', _data.head())

    positives = _data.loc[_data.label == 1]
    negatives = _data.loc[_data.label == 0]
    print('\n\n', 'Positive examples: {}  Negative examples: {}'.format(len(positives), len(negatives)))

    train_data, test_data = split_data(_data)

    train_input_examples = convert_data_into_input_examples(train_data)
    test_input_examples = convert_data_into_input_examples(test_data)

    bert_train_data = tokenize(train_input_examples)
    bert_test_data = tokenize(test_input_examples)

    _model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
    compile_model(_model)

    # TODO: fix workaround
    x_train, y_train = my_solution(bert_train_data)
    x_test, y_test = my_solution(bert_test_data)

    print('x_train shape: {}'.format(x_train[0].shape))
    print('x_val shape: {}'.format(x_test[0].shape))

    train_ds = tf.data.Dataset.from_tensor_slices((x_train[0], x_train[1], x_train[2], y_train)).map(
        example_to_features).shuffle(100).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test[0], x_test[1], x_test[2], y_test)).map(example_to_features).batch(
        64)

    print('Format of model input examples: {} '.format(train_ds.take(1)))

    history = _model.fit(train_ds, validation_data=test_ds, epochs=values.epochs)
    # plot_learning_process(history)
