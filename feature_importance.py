import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score

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
        



    def importances_target(self, X, y_true, y_pred, type="feat", cascade=False):

        """Calculate the permutation importance

        @param X: Input DataFrame or 2D numpy array.
        @param y_true: true target values
        @param y_pred: predictions for the target values
        @param type: type of feature importance, either normal feature importance or permutation feature importance
        @param cascade: True (if we apply cascading effect) or False
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        model_list = self.models.copy()
        feat_imp_score = []

        # Calculate (normal) feature importance
        if type == "feat":
            for j in range(len(y_pred.columns)):
                score = model_list[j].feature_importances_
                #score = pd.Series(score, index=list(X.columns)+list(y_pred.columns[:j+1])[:-1])
                feat_imp_score.append(score)

        elif type == "perm":
            data = pd.concat([X, y_pred], axis=1)
            v = X.shape[1] # number of features

            if cascade == False: 
                for a in range(1, y_pred.shape[1] + 1):  
                    y_true_a = y_true.iloc[:, a-1]  
                    y_pred_a = y_pred.iloc[:, a-1]  
                    feat_imp_score_model = []
                    """
                    TO DO: make code more clean (checking type=object outside for loop for all predictors)
                    """
                    # Calculate importance of each predictor 
                    for i in range(v+a-1):  
                        Xext = data.iloc[:, :a+v-1].copy()  
                        last = Xext.iloc[:, -1]
                        # Convert categorical to numerical encoding
                        if last.dtype == 'object':
                            yi_pred_num = pd.Categorical(last).codes
                            yi_pred_num = pd.Series(yi_pred_num)
                            Xext.iloc[:, -1] = yi_pred_num
                        # For the input variables from X, don't use cascading effect
                        # Shuffle feature i in Xext + predict only model a
                        Xext.iloc[:, i] = Xext.iloc[:, i].sample(frac=1).reset_index(drop=True)
                        yi_pred_shuffled = model_list[a-1].predict(Xext)
                    
                        # Remove rows with missing values in 
                        assert len(y_true_a) == len(yi_pred_shuffled)
                        missing_rows_mask = pd.Series(y_true_a).notna()  
                        y_true = y_true_a[missing_rows_mask]
                        y_pred_actual = y_pred_a[missing_rows_mask]
                        y_pred_shuffled_actual = yi_pred_shuffled[missing_rows_mask]

                        # Calculate evaluation metrics (R2 or accuracy)
                        if y_true.dtype.kind in 'bifc':
                            score1 = r2_score(y_true, y_pred_actual)
                            score2 = r2_score(y_true, y_pred_shuffled_actual)
                        else:
                            score1 = accuracy_score(y_true, y_pred_actual)
                            score2 = accuracy_score(y_true, y_pred_shuffled_actual) 

                        # Obtain permutation feature importance score for the input of model a
                        feat_imp_score_model.append(score1-score2)
                    
                    # Store all permutation importance lists for each model in one big list
                    feat_imp_score.append(feat_imp_score_model)

            # Apply cascading effect
            elif cascade == True: 
                for a in range(1, y_pred.shape[1] + 1):  
                    y_true_a = y_true.iloc[:, a-1]  
                    y_pred_a = y_pred.iloc[:, a-1]  
                    feat_imp_score_model = []
                    """
                    TO DO: make code more clean (checking type=object outside for loop for all predictors)
                    """
                    # Calculate importance of each predictor 
                    for i in range(v+a-1): 
                        Xext = data.iloc[:, :a+v-1].copy()  
                        last = Xext.iloc[:, -1]
                        # Convert categorical to numerical encoding
                        if last.dtype == 'object':
                            yi_pred_num = pd.Categorical(last).codes
                            yi_pred_num = pd.Series(yi_pred_num)
                            Xext.iloc[:, -1] = yi_pred_num 
                        # Shuffle
                        Xext.iloc[:, i] = Xext.iloc[:, i].sample(frac=1).reset_index(drop=True)
                        Xext_ori = Xext.copy()
                        last = Xext.iloc[:, -1]
                        # Convert categorical to numerical encoding
                        if last.dtype == 'object':
                            yi_pred_num = pd.Categorical(last).codes
                            yi_pred_num = pd.Series(yi_pred_num)
                            Xext.iloc[:, -1] = yi_pred_num
                        for m in range(max(-1, i-v), a):  
                            yi_pred = model_list[m+1].predict(Xext)
                            # Convert categorical to numerical encoding
                            if yi_pred.dtype == 'object':
                                yi_pred_num = pd.Categorical(yi_pred).codes
                                yi_pred_num = pd.Series(yi_pred_num)
                                column_name = data.iloc[:, m+v+1].name
                                Xext[column_name] = yi_pred_num
                            else:
                                column_name = data.iloc[:, m+v+1].name
                                Xext[column_name] = yi_pred
                            Xext_ori[column_name] =  yi_pred
                        yi_pred_shuffled = Xext_ori.iloc[:, -1]

                        assert len(y_true_a) == len(yi_pred_shuffled)
                        missing_rows_mask = pd.Series(y_true_a).notna()  
                        y_true = y_true_a[missing_rows_mask]
                        y_pred_actual = y_pred_a[missing_rows_mask]
                        y_pred_shuffled_actual = yi_pred_shuffled[missing_rows_mask]

                        # Calculate evaluation metrics (R2 or accuracy)
                        if y_true.dtype.kind in 'bifc':
                            score1 = r2_score(y_true, y_pred_actual)
                            score2 = r2_score(y_true, y_pred_shuffled_actual)
                        else:
                            score1 = accuracy_score(y_true, y_pred_actual)
                            score2 = accuracy_score(y_true, y_pred_shuffled_actual) 

                        # Obtain permutation feature importance score for the input of model a
                        feat_imp_score_model.append(score1-score2)
                        
                    # Store all permutation importance lists for each model in one big list
                    feat_imp_score.append(feat_imp_score_model)

        return feat_imp_score
