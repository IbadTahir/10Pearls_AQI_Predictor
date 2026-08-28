from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

HORIZONS = [24, 48, 72]


class MultiHorizonXGBoost:
    
    def __init__(self, estimators: Optional[Dict[int, XGBRegressor]] = None):
        self.estimators_ = estimators or {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.DataFrame] = None):
        self.estimators_ = {}
        for idx, h in enumerate(HORIZONS):
            col = f"target_{h}h"
            y_col_train = y_train[col]

            max_depth = 6 if h == 24 else (7 if h == 48 else 7)
            subsample = 0.85
            colsample = 0.80

            model = XGBRegressor(
                n_estimators=450,
                max_depth=max_depth,
                learning_rate=0.03,
                subsample=subsample,
                colsample_bytree=colsample,
                min_child_weight=4,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42 + h,
                n_jobs=-1,
            )

            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_col_train), (X_val, y_val[col])]
                model.fit(
                    X_train,
                    y_col_train,
                    eval_set=eval_set,
                    verbose=False,
                )
            else:
                model.fit(X_train, y_col_train)

            self.estimators_[h] = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for h in HORIZONS:
            col_pred = self.estimators_[h].predict(X)
            preds.append(col_pred)
        return np.column_stack(preds)

