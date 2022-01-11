data_files_dir = 'data'
unprocessed_data_file_name = 'data-en-hi-de-fr.csv'
processed_data_file_name = 'data.csv'

percent_of_data_used = 0.1
data_test_to_train_ratio = 0.2
MAX_SEQ_LENGTH = 512
labels_list = [0, 1]

train_batch_size = 16
test_batch_size = 64

# optimizer
learning_rate = 2e-5
epsilon = 1e-08
clipnorm = 1.0

epochs = 3

languages = ['en', 'hi', 'de', 'fr']
convert_langs_vals = {
    'en': {
        'column_idx': 2,
        'drop': None
    },
    'hi': {
        'column_idx': 3,
        'drop': ['text']
    },
    'de': {
        'column_idx': 4,
        'drop': ['text', 'text_hi']
    },
    'fr': {
        'column_idx': 5,
        'drop': ['text', 'text_hi', 'text_de']
    },
}
