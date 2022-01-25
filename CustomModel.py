import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, f1_score

class Model():
    def __init__(self, model_params: dict):
        self.model = None
        self.model_params = model_params

    def train(self, dataset_path: str, save_model_path: str = None):
        self.model = fasttext.train_supervised(input=dataset_path, **self.model_params)
        if save_model_path is not None:
            self.model.save_model(save_model_path)

    
    def accuracy(self, y_true, y_pred):
        return np.sum(y_pred == y_true) / len(y_true)

    def predict(self, text: str = None, texts: pd.Series = None):
        assert text is None or texts is None
        if texts is not None:
            preds = texts.apply(lambda x: self.model.predict(x, k=1)[0][0])
            # preds = []
            # for text in texts:
            #     preds.append(self.model.predict(text, k=1)[0][0])
            return preds

        return self.model.predict(text, k=1)[0][0]

    def calc_f1(self, test_dataset_path):
        res = self.model.test(test_dataset_path)
        percision, recall = res[1], res[2]
        return 2 * (percision * recall) / (percision + recall)

    def calc_f1(self, y_true, y_pred, avg):
        return f1_score(y_true, y_pred, average=avg)
    
    def accuracy_per_ind(self, ind, y_true, y_pred):
        y_true_ind = (y_true == ind)
        y_pred_ind = (y_pred == ind)
        return self.accuracy(y_true_ind, y_pred_ind)
    
    def precision_per_ind(self, ind, y_true, y_pred):
        y_true_ind = (y_true == ind)
        y_pred_ind = (y_pred == ind)
        return precision_score(y_true_ind, y_pred_ind)


    def sensitivity_per_ind(self, ind, y_true, y_pred):
        y_true_ind = (y_true == ind)
        y_pred_ind = (y_pred == ind)
        TP = ((y_pred_ind)*(y_true_ind)).sum()
        FN = ((~y_pred_ind)*(y_true_ind)).sum()
        return TP / (TP + FN)
    

    
    