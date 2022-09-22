import pandas as pd
import numpy as np
import torch 
from torch.utils import data
from torch import nn

from typing import List, Tuple, Dict, Union, Optional
from typing import Iterable
import surprise
from .models import AutoRec, Model, CollobarativeModel


class Dataset(data.Dataset):
    
    def __init__(
        self,
        ratings: pd.DataFrame,
        users_cnt: int,
        items_cnt: int,
        user_based: bool = True,
        long_matrix: bool = True,
        user_column: str = "user_idx",
        item_column: str = "item_idx",
        rating_column: str = "rating_idx"
    ) -> None:
        super().__init__()
        self.data = ratings
        self.ui_shape = (users_cnt, items_cnt)
        self.user_based = user_based
        
        
        if long_matrix:
            self.data = self.data.pivot(index=user_column, columns=item_column, values=rating_column)
            if self.data.shape != self.ui_shape:
                self.data = np.zeros(self.ui_shape)
                for idx in ratings.index:
                    self.data[ratings.loc[idx, user_column]][ratings.loc[idx, item_column]]=ratings.loc[idx, rating_column]
                
        self.data = np.array(self.data, dtype=float)
        self.data = np.nan_to_num(self.data, copy=True, nan=0.)
        
        if not self.user_based:
            self.data = self.data.T
        
        self.data = torch.from_numpy(self.data).float()
        
    def __getitem__(
        self, 
        index
    ) -> Tuple[torch.Tensor]:
        return self.data[index]
    
    def __len__(
        self
    ) -> int:
        if self.user_based:
            return self.ui_shape[0]
        else:
            return self.ui_shape[1]
        

        

def svd(train, test, sample, cnt=10, with_null=False):
    rmse=[]
    mae=[]
    ndcg=[]
    for i in range(cnt):
        svd = CollobarativeModel(surprise.SVD, "SVD")
        svd.train(train[i])
        errors = svd.test(test[i], with_null)
        
        rmse.append(errors["rmse"])
        mae.append(errors["mae"])
        ndcg.append(errors["ndcg"])
        
    return pd.DataFrame({"model": "SVD", "sample_size": sample, "rmse": rmse, "mae": mae, "ndcg": ndcg})


def nmf(train, test, sample, cnt=10, with_null=False):
    rmse=[]
    mae=[]
    ndcg=[]
    for i in range(cnt):
        svd = CollobarativeModel(surprise.NMF, "SVD")
        svd.train(train[i])
        errors = svd.test(test[i], with_null)
        
        rmse.append(errors["rmse"])
        mae.append(errors["mae"])
        ndcg.append(errors["ndcg"])
        
    return pd.DataFrame({"model": "NMF", "sample_size": sample, "rmse": rmse, "mae": mae, "ndcg": ndcg})

def knn(train, test, sample, cnt=10, with_null=False):
    rmse=[]
    mae=[]
    ndcg=[]
    for i in range(cnt):
        svd = CollobarativeModel(surprise.KNNBasic, "kNN")
        svd.train(train[i])
        errors = svd.test(test[i], with_null)
        
        rmse.append(errors["rmse"])
        mae.append(errors["mae"])
        ndcg.append(errors["ndcg"])
        
    return pd.DataFrame({"model": "kNN", "sample_size": sample, "rmse": rmse, "mae": mae, "ndcg": ndcg})


def autorec(train, test, validation_data, sample, cnt=10, with_null=False):
    rmse=[]
    mae=[]
    ndcg=[]
    for i in range(cnt):
        i_autorec = AutoRec(
            input_size=validation_data.item_id.unique().shape[0],
            hidden_dims=[512],
            encoder_activation_fn = nn.Sigmoid,
            decoder_activation_fn = None,
            dropout=0.05,
            bias=True
        )
        
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(0.01)
                
        i_autorec.apply(init_weights)
        model = Model(
            model=i_autorec,
            optimizer=torch.optim.Adam,
            optimizer_config={"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 0.01},
            loss_fn=nn.MSELoss(),
            batch_size=16,
            num_epoch=50,
            device=torch.device("cuda:1")
        )
        _ = model.train(train[i], 0.1, False)
        errors = model.test(test[i])
        #errors = model.test(train[i], test[i], with_null)
        
        rmse.append(errors["rmse"])
        mae.append(errors["mae"])
        ndcg.append(errors["ndcg"])
        
    return pd.DataFrame({"model": "AutoRec", "sample_size": sample, "rmse": rmse, "mae": mae, "ndcg": ndcg})