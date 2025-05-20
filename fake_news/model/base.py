from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from fake_news.utils.features import Datapoint


class Model(ABC):
    @abstractmethod
    def train(self,
              train_datapoints: List[Datapoint],
              val_datapoints: List[Datapoint],
              cache_featurizer: Optional[bool] = False) -> None:
        """
        Trains the model using provided data.
        
        Parameters:
            train_datapoints: Training dataset
            val_datapoints: Validation dataset for model tuning
            cache_featurizer: If True, caches the featurizer to improve performance
            
        Return:
            None
        """
        pass
    
    @abstractmethod
    def predict(self, datapoints: List[Datapoint]) -> np.array:
        """
        Performs inference of model on collection of datapoints. Returns an
        array of model predictions. 
        
        Parameters:
            datapoints: List of datapoints to perform inference on
        
        Return: 
            Array of predictions
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, eval_datapoints: List[Datapoint], split: Optional[str] = None) -> Dict:
        """
        Compute a set of model-specifc metrics on the provided set of datapoints.
        
        Parameters:
            eval_datapoints: Datapoints to compute metrics for
            split: Data split on which metrics are being computed
        
        Return: 
            A dictionary mapping from the name of the metric to its value
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """
        Return the model-specific parameters such as number of hidden-units in the case
        of a neural network 
        
        Returns:
            Dictionary containing the model parameters
        """
        pass
