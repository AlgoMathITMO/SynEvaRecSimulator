import pandas as pd
import numpy as np
import torch 
from torch.utils import data

from typing import List, Tuple, Dict, Union, Optional
from typing import Iterable

from torch import nn
from torch.utils.data import DataLoader
import scipy
import surprise

from sklearn.metrics import ndcg_score
def ndcg_fn(ground_truth, prediction):
    if len(ground_truth) == 1 or len(ground_truth) == 0:
        return np.nan
    return ndcg_score(np.array([ground_truth]), np.array([prediction]))
ndcg_vect = np.vectorize(ndcg_fn)

class AutoRec(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dims: Union[int, List[int]],
        encoder_activation_fn: Optional[type] = nn.Sigmoid,
        decoder_activation_fn: Optional[type] = None,
        dropout: float = 0.,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        sizes = [input_size] + hidden_dims
        layers = []
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(sizes[i-1], sizes[i], bias=bias))
            if encoder_activation_fn:
                layers.append(encoder_activation_fn())
            if dropout != 0.:
                layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        
        sizes = sizes[::-1]
        layers = []
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(sizes[i-1], sizes[i], bias=bias))
            if decoder_activation_fn:
                layers.append(decoder_activation_fn())
            if dropout != 0.:
                layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, input):
        z = self.encoder(input)
        prediction = self.decoder(z)
        
        if self.training:
            prediction = prediction*input.abs().sign()
        
        return prediction
    
    
class Model:
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: type,
        optimizer_config: Dict,
        loss_fn: nn.Module,
        batch_size: int,
        num_epoch: int,
        device: torch.device = torch.device("cpu"),
        non_zeros: bool = False
    ) -> None:
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optimizer_config)
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.device = device
        
    def train(
        self,
        train_data: data.Dataset,
        validation: Optional[float] = None,
        is_printed: bool = False
    ) -> Tuple[List[float]]:
        if validation:
            validation_idx = [i for i in range(len(train_data)) if np.random.rand() < validation]
            train_idx = [i for i in range(len(train_data))]
            train_idx = list(set(train_idx).difference(set(validation_idx)))
            validation_data = train_data[validation_idx]
            train_data = train_data[train_idx]
            
        train_loader = DataLoader(train_data, self.batch_size, shuffle=True)
        
        self.model.to(self.device)
        self.model.train()
        
        train_loss = []
        validation_loss = []
        
        for epoch in range(self.num_epoch):
            loss_arr = []
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.loss_fn(batch, output)
                loss.backward()
                self.optimizer.step()
                
                loss_arr.append(loss.sqrt().item())
            
            train_loss.append(np.mean(loss_arr))
            message = f"Epoch: {epoch}, train RMSE: {train_loss[-1]}"
            if validation:
                test_loss = self.test(validation_data)
                validation_loss.append(test_loss["rmse"])
                message = message + f", validation RMSE: {validation_loss[-1]}"
                
            if is_printed:
                print(message)
        
        return train_loss, validation_loss
    
    def test(self, 
             test_data: data.Dataset,
             ground_truth: Optional[data.Dataset]=None,
             with_nulls: bool = False
    ) -> Dict[str, int]:
        
        cnt = 0
        rmse = 0
        mae = 0
        ndcg = []
        
        self.model.to(self.device)
        self.model.eval() 
        
        input_data = test_data
        gt_data = ground_truth if ground_truth else input_data
        mask_data_input = input_data if ground_truth else None
        mask_data_ground_truth = gt_data 
        
        with torch.no_grad():
            for idx in range(len(input_data)):
                input_batch = input_data[idx].to(self.device)
                gt_batch = gt_data[idx]
                mask_input = torch.nonzero(1-mask_data_input[idx].sign(), as_tuple=True) if ground_truth else None
                mask_gt = torch.nonzero(mask_data_ground_truth[idx], as_tuple=True)
                output_batch = self.model(input_batch).cpu()
                
                if mask_input:
                    gt_batch_ = gt_batch[mask_input]
                    output_batch_ = output_batch[mask_input]
                else:
                    gt_batch_ = gt_batch[mask_gt]
                    output_batch_ = output_batch[mask_gt]
                    
                gt_batch = gt_batch[mask_gt]
                output_batch = output_batch[mask_gt]
                
                if with_nulls:
                    gt_batch = gt_batch_
                    output_batch = output_batch_
                
                cnt += len(output_batch)
                rmse += ((output_batch - gt_batch) ** 2).sum().item()
                mae += (output_batch - gt_batch).abs().sum().item()
                ndcg.append(ndcg_fn(gt_batch_.numpy(), output_batch_.numpy()))
                
        if cnt != 0:
            result = {"rmse":np.sqrt(rmse/cnt), "mae": mae/cnt, "ndcg":np.nanmean(ndcg)}
        else:
            result = {"rmse":-1, "mae": -1, "ndcg":np.nanmean(ndcg)}
        return result
                
    def __call__(
        self, 
        data
    ):
        self.model.to(self.device)
        self.model.eval()
        
        data = data.to(self.device)
        output = self.model(data)
        
        return output
    
    

def _rating_matrix_to_long_table(
    rating_matrix: Union[pd.DataFrame, np.ndarray],            
    drop_none: bool = True,
    user_column: str = "user_idx",
    item_column: str = "item_idx",
    rating_column: str = "rating_idx"
) -> pd.DataFrame:
    df = pd.DataFrame(rating_matrix)
    df["user_id"] = df.index
    df_long = df.melt(id_vars=[user_column], var_name=item_column, value_name=rating_column).copy()
    if drop_none:
        df_long.drop(df_long[df_long.rating.isna()].index, inplace=True)
    return df_long

class CollobarativeModel():
    
    def __init__(
        self,
        model: type,
        model_name: str
    ) -> None:
        self.model=model()
        self.model_name=model_name
        
    def train(
        self,
        train_data: data.Dataset,
    ) -> Tuple[List[float]]:
        train_data = _rating_matrix_to_long_table(train_data[:].numpy(), False, "user_id", "item_id", "rating")
        self.df_train = train_data.copy()
        
        
        train_data = surprise.Dataset.load_from_df(train_data[['user_id', 'item_id', 'rating']][train_data.rating!=0], surprise.Reader(rating_scale=(1, 10))).build_full_trainset()
        print(train_data)
        self.model.fit(train_data)
        
    def test(
        self,
        test_data: data.Dataset,
        with_null: bool =False
    ) -> Dict[str, int]:
        test_data = _rating_matrix_to_long_table(test_data[:].numpy(), False, "user_id", "item_id", "rating")
        test_data = test_data.drop(self.df_train[self.df_train.rating!=0].index)
        
        df_test_short = surprise.Dataset.load_from_df(test_data[['user_id', 'item_id', 'rating']][test_data.rating!=0], surprise.Reader(rating_scale=(1, 10)))
        _, df_test_short = surprise.model_selection.train_test_split(df_test_short, test_size=1.)

        df_test_long = surprise.Dataset.load_from_df(test_data[['user_id', 'item_id', 'rating']], surprise.Reader(rating_scale=(1, 10)))
        _, df_test_long = surprise.model_selection.train_test_split(df_test_long, test_size=1.)
        
        predictions_1 = self.model.test(df_test_short)
        
        predictions_2 = self.model.test(df_test_long)

        predictions_df = pd.DataFrame(np.array(list(map(lambda x: [x.uid, x.iid, x.r_ui, x.est], predictions_2))), columns=["user_idx","item_idx", "relevance", "prediction"])
        predictions_df = predictions_df.groupby("user_idx")[["relevance", "prediction"]].agg(pd.Series.tolist)
        
        ndcg=np.nanmean(ndcg_vect(predictions_df.relevance, 
                                  predictions_df.prediction
                                 )
                       )
        
        if with_null:
            rmse = surprise.accuracy.rmse(predictions_2)
            mae = surprise.accuracy.mae(predictions_2)
        else:
            rmse = surprise.accuracy.rmse(predictions_1)
            mae = surprise.accuracy.mae(predictions_1)
            
        result = {"rmse": rmse, "mae":mae, "ndcg": ndcg}
        return result