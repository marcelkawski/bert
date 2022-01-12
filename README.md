# bert
Multilingual BERT algorithm for spam detection on text data

Uruchomienie następuje przy pomocy pliku bert.py z parametrami:
- percent_of_data_used - procent danych użytych w skrypcie
- data_test_to_train_ratio - stosunek zbioru walidacyjnego do trenującego
- MAX_SEQ_LENGTH - maksymalna długość pojedynczej sentencji
- train_batch_size - rozmiar partii trenującej
- test_batch_size - rozmiar partii walidacyjnej
- epochs - ilość epok


Przykładowe uruchomienie:

<i>python bert.py --data_used 0.5 --data_ratio 0.1 --max_seq 500 --train_batch 14 --test_batch 60 --epochs 2</i>
