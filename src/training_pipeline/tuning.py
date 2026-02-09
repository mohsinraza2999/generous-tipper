import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.utils import results_data
from src.utils.evaluate import EvaluateMetrices
from src.utils.logging import logger
from typing import Dict
class Tune(EvaluateMetrices):
    def __init__(self, pipeline, model_name, params: Dict, scoring: Dict, target_matrix: str):
        self.model_name=model_name
        self.target_matrix=target_matrix
        # "n_jobs=-1" parallelization and resourse control
        self.tune_tipper=GridSearchCV(pipeline, params, scoring=scoring,
                                      cv=5,refit=self.target_matrix, n_jobs=-1)
        (logger.info if logger else print)("Instantiate the GridSearchCV Object")

    def gride_search_tuning(self, X_train, y_train, X_val, y_val):

        
        

        (logger.info if logger else print)(f"{self.model_name} Model H_P Fine Tuning Starts")
        self.tune_tipper.fit(X_train,y_train)
        (logger.info if logger else print)(f"{self.model_name} Model H_P Fine Tuned")

        results=self.make_results(self.model_name, self.tune_tipper, self.target_matrix)

        preds=self.tune_tipper.predict(X_val)
        (logger.info if logger else print)(f"Conduct Test on {self.model_name} H_P Model")
        test_scores=self.get_test_scores(self.model_name+' Test', preds, y_val)

        results = pd.concat([results, test_scores], axis=0,ignore_index=True)

        results_data.save_results(results)
        (logger.info if logger else print)(f"Train and Test Results are Saved")

        results_data.save_best(self.tune_tipper.best_estimator_, self.model_name,
                               self.tune_tipper.best_params_, self.tune_tipper.best_score_)
        (logger.info if logger else print)(f"{self.model_name} H_P Model, H_P, and Score are Saved")


        return {"model_name": self.model_name,
                "best_score": self.tune_tipper.best_score_}

