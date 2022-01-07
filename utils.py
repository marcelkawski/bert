import os

import values


def get_unproc_data_file_name():
    return os.path.join(values.data_files_dir, values.unprocessed_data_file_name)


def get_data_file_name():
    return os.path.join(values.data_files_dir, values.processed_data_file_name)
