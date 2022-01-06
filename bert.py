import pandas as pd
import tensorflow as tf
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

    _model.fit(bert_train_data, validation_data=bert_test_data, epochs=3)
