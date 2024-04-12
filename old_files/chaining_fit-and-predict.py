import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class Chain(BaseEstimator):
    """Regression and classification chains for multi-target prediction."""

    def __init__(self, model_reg, model_clf, propagate="pred"):
        """Initializes this chain.
        
        @param model_reg: The base model to use for regression targets.
        @param model_clf: The base model to use for classification targets.
        @param propagate: Whether to give predictions ("pred") or true values 
            ("true") as extra input features to the next model in the chain. Put
            `False` for training without chaining instead (i.e., local method).
        """
        self.base_model = {
            "reg": model_reg,
            "clf": model_clf,
        }
        self.propagate = propagate

    def fit(self, X, y, target_types=None):
        """Fit this chain to the given data.

        @param X: Input DataFrame or 2D numpy array.
        @param y: Output DataFrame or 2D numpy array (in chaining order).
        @param target_types: List defining the type of each of the columns of y.
            "clf" means a classification target, "reg" a regression target.
        """
        # Ensure we are working with dataframes
        # (some sklearn preprocessing transforms into numpy arrays instead)
        # (and ensure that y.columns has no overlap with X.columns)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=[f"y{i}" for i in range(y.shape[1])])
        assert all(X.index == y.index)
        self.y_columns = y.columns # needed for self.predict
        # Infer target types if not given NOTE temporary hack!
        if target_types is None:
            target_types = ["clf" if y[target].nunique() < 10 else "reg"
                            for target in y.columns]
        else:
            assert len(target_types) == len(y.columns)

        # Start the training loop
        self.models = [copy.deepcopy(self.base_model[target])
                       for target in target_types]
        Xext = X.copy() # X extended with predictions for each target
        for col, model in zip(self.y_columns, self.models):
            # Fit model on only observed target values
            i_nona = ~y[col].isna()
            model.fit(Xext.loc[i_nona,:], y[col].loc[i_nona]) # Fit on fully annotated (target side) part
            # Propagate the predictions
            if self.propagate == "pred":
                yi_pred = model.predict(Xext)
                if yi_pred.dtype == 'object':
                    yi_pred = pd.Categorical(yi_pred).codes
                    yi_pred = pd.Series(yi_pred).replace(-1, np.nan, inplace=True)
                Xext[col] = yi_pred
            elif self.propagate == "true":
                yi_true = y[col]
                if yi_true.dtype == 'object':
                    yi_true = pd.Categorical(yi_true).codes
                    yi_true = pd.Series(yi_true).replace(-1, np.nan, inplace=True)
                Xext[col] = yi_true
        return self

    def predict(self, X):
        """Use the chain to predict for new data.

        @param X: Input DataFrame or 2D numpy array.
        """
        assert len(self.y_columns) == len(self.models)  # 1 model for each target
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        Xext = X.copy()
        pred_list = X.copy()
        for col, model in zip(self.y_columns, self.models):
            if self.propagate == False:
                yi_pred = model.predict(X)
            else:
                yi_pred = model.predict(Xext)
                if yi_pred.dtype == 'object':
                    yi_pred_num = pd.Categorical(yi_pred).codes
                    yi_pred_num = pd.Series(yi_pred_num).replace(-1, np.nan, inplace=True)
                    Xext[col] = yi_pred_num
                else: 
                    Xext[col] = yi_pred
            pred_list[col] = yi_pred
        return pred_list.iloc[:, -len(self.models):]