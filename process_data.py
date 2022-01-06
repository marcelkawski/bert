import pandas as pd

import utils
import values
from values import convert_langs_vals as clv


def convert(data, lang_str):
    data = data.iloc[:, :clv[lang_str]['column_idx']]
    drop = clv[lang_str]['drop']
    if drop is not None:
        data = data.drop(drop, axis=1)
    data = data.replace(to_replace='ham', value=0).replace(to_replace='spam', value=1)
    data.columns.values[1] = 'text'
    data['lang'] = lang_str
    return data


def convert_all():
    unproc_data_file_path = utils.get_unproc_data_file_name()
    data = pd.read_csv(unproc_data_file_path)

    dfs = [convert(data, lang_str) for lang_str in values.languages]
    data = pd.concat(dfs)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.rename({'labels': 'label'}, axis=1)

    data_file_path = utils.get_data_file_name()
    data.to_csv(data_file_path, mode='w', index=False, header=True)


if __name__ == "__main__":
    convert_all()
