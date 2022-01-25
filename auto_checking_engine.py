import pandas as pd
import numpy as np
from CustomModel import Model
import csv
import random
import torch
import sys


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


def train_model(train_df, fasttext_train_file_path, save_model_path=None):
    print("started writing train file")
    train_df.to_csv(fasttext_train_file_path,
                    index=False,
                    sep=' ',
                    header=None,
                    quoting=csv.QUOTE_NONE,
                    quotechar="",
                    escapechar=" ")

    model_params = {'dim': 90, 'epoch': 14, 'lr': 0.05, 'lrUpdateRate': 100, 'maxn': 0, 'minCount': 1, 'minn': 0,
                    'neg': 5, 't': 0.0001, 'thread': 12, 'verbose': 2, 'ws': 5}
    model = Model(model_params)
    print('started training')
    model.train(fasttext_train_file_path, save_model_path=save_model_path)
    print('finished training')

    return model


def evaluate_model(model, test_df):
    y_pred = model.predict(texts=test_df['text'])
    f1_dict = {}
    f1_dict['micro'] = model.calc_f1(test_df['industry'], y_pred, 'micro')
    f1_dict['macro'] = model.calc_f1(test_df['industry'], y_pred, 'macro')
    f1_dict['weighted'] = model.calc_f1(test_df['industry'], y_pred, 'weighted')

    for avg, f1 in f1_dict.items():
        print(f"f1_{avg}: ", round(f1, 3))

    sensitivities = {}
    for industry in sorted(test_df['industry'].unique()):
        sensitivity_per_ind = model.sensitivity_per_ind(industry, test_df['industry'], y_pred)
        print("Sensetivity for", industry, ':', round(sensitivity_per_ind, 3))
        sensitivities[industry] = sensitivity_per_ind

    return f1_dict, sensitivities


if __name__ == "__main__":
    set_seed()
    try:
        fasttext_train_file_path = sys.argv[1]
        save_model_path = sys.argv[2]
        train_dataset_path = sys.argv[3]
        test_dataset_path = sys.argv[4]
        ids = sys.argv[5]
    except IndexError:
        print("Not enough arguments given to the auto checking engine")
        exit(-1)
    frame = pd.read_csv(train_dataset_path)
    test_df = pd.read_csv(test_dataset_path)
    model = train_model(frame, fasttext_train_file_path, save_model_path=save_model_path)
    f1, sensitivities = evaluate_model(model, test_df)
    print(f"Average recall per industry for the submission of {ids}: {np.mean(sensitivities.values())}")