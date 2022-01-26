import csv
import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from langdetect import detect
import sys

csv.field_size_limit(sys.maxsize)

class DatasetPreprocess:
    def __init__(self, dataset_path=None, processed_dataset_path=None, train_size=0.85, ind_tokenizer_path=None, lemmatizer=None,
                 remove_words=None, write_fields=('industry', 'text', 'lang'), start_trim_sentences=1, end_trim_sentences=2,
                 min_sentences=3):
        self.dataset_path = dataset_path
        self.lemmatizer = lemmatizer
        self.processed_dataset_path = processed_dataset_path
        self.write_fields = write_fields
        self.ind_tokenizer = None
        self.train_size = train_size
        self.remove_words = remove_words
        self.start_trim_sentences = start_trim_sentences
        self.end_trim_sentences = end_trim_sentences
        self.min_sentences = min_sentences
        if ind_tokenizer_path is not None:
            self.ind_tokenizer = self.get_ind_tokenizer(ind_tokenizer_path)

    @staticmethod
    def predict_lang(text):
        try:
            lang = detect(text)
        except:
            print(text)
            lang = None
        return lang

    @staticmethod
    def get_ind_tokenizer(ind_tokenizer_path: str):
        with open(ind_tokenizer_path, 'rb') as f:
            d = pickle.load(f)
        return d

    @staticmethod
    def clean_text(s, lemmatizer=None, remove_words=None):
        s = re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', s)

        s = re.sub(r"[^A-Za-z \(\)\-.:',?&0-9]+", ' ', s)
        # reduce multiple spaces
        s = re.sub(' +', ' ', s)
        # reduce multiple dots
        s = re.sub('\.+', '.', s)
        s = re.sub(' w ', ' ', s)

        if lemmatizer is not None:
            new_s = []
            for w in s.split():
                new_s.append(lemmatizer.lemmatize(w))
            s = ' '.join(new_s)

        # Removing bad words
        if remove_words is not None:
            s = ' '.join(filter(lambda x: x not in remove_words, s.split()))
        return s

    @staticmethod
    def choose_part_of_text(text, start_trim_sentences, end_trim_sentences, min_sentences):
        sentences = text.split('.')
        keep_sentences = sentences[start_trim_sentences: -end_trim_sentences]
        if 0 < len(keep_sentences) < min_sentences:
            keep_sentences = sentences[1: -1]
        elif len(keep_sentences) == 0:
            keep_sentences = sentences
        return '.'.join(keep_sentences)

    def preprocess(self):
        with open(self.dataset_path, mode='r') as csvreadfile, open(self.processed_dataset_path, newline='',
                                                                    mode='w') as csvwritefile:
            reader = csv.DictReader(line.replace('\0', '') for line in csvreadfile)
            writer = csv.DictWriter(csvwritefile, fieldnames=self.write_fields)
            writer.writeheader()
            for row in tqdm(reader):
                processed_text = self.clean_text(row['text'], self.lemmatizer, self.remove_words)
                lang = self.predict_lang(processed_text)
                
                processed_text = self.choose_part_of_text(processed_text, self.start_trim_sentences,
                                                          self.end_trim_sentences, self.min_sentences)
                industry = self.ind_tokenizer[row['industry']] if self.ind_tokenizer is not None else row['industry']
                row_dict = {'industry': '__label__' + str(industry), 'text': processed_text, 'lang': lang}
                writer.writerow(row_dict)

            writer.writerow(row_dict)

    def write_fasttext_dataset(self, fasttext_train_file_path, fasttext_test_file_path, dataset=None):
        assert self.processed_dataset_path is not None or dataset is not None
        dataset = pd.read_csv(self.processed_dataset_path).fillna("") if dataset is None else dataset

        split_msk = np.random.rand(len(dataset)) < self.train_size
        train_ds = dataset[split_msk]
        val_ds = dataset[~split_msk]
        print("train size:", len(train_ds), "test size:", len(val_ds))

        del dataset

        train_ds.to_csv(fasttext_train_file_path,
                        index=False,
                        sep=' ',
                        header=None,
                        quoting=csv.QUOTE_NONE,
                        quotechar="",
                        escapechar=" ")

        val_ds.to_csv(fasttext_test_file_path,
                      index=False,
                      sep=' ',
                      header=None,
                      quoting=csv.QUOTE_NONE,
                      quotechar="",
                      escapechar=" ")

if __name__ == '__main__':
    for chunk in [0,1,2]:
        data_path = f"/datashare/2021/data_chunk_{chunk}.csv"
        write_path = f"/StudentData/Project/data_chunk_{chunk}.csv"
        print(write_path)
        ind_tokenizer_path = "/StudentData/Project/ind_tokenizer.pkl"
        data_proccesor = DatasetPreprocess(data_path, write_path, ind_tokenizer_path=ind_tokenizer_path)
        data_proccesor.preprocess()
