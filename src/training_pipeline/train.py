from joblib import load, dump
from sklearn.pipeline import Pipeline
from pathlib import Path
import json
import pandas as pd
#from src.utils import load_config
from src.data_pipeline import dataset
from src.model_pipeline.tipper_models import TipperModels
from src.training_pipeline.tuning import Tune
from src.utils.configs import Configurations
from src.utils.logging import logger


class Train(Tune):
    def __init__(self, model, model_name, params, scoring, target_matrix):
        super().__init__(model, model_name, params, scoring, target_matrix)

    def tune(self, X_train, y_train, X_val, y_val):
        return self.gride_search_tuning(X_train, y_train, X_val, y_val)
    
"""    def train_on_best(self, model_type, best_params, X_train, y_train):
        model = self.model_builder(model_type,best_params)
        model.fit(X_train, y_train)"""


def pipeline_builder(model_type):
    tipper_object=TipperModels(model_type)
    (logger.info if logger else print)(f"{model_type} model loaded")
    
    config_object=Configurations()
    config=config_object.data_configs()
    
    preprocessor= dataset.data_pipeline(config['categorical'],config['numerical'])
    (logger.info if logger else print)("Data Pipeline Build")

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", tipper_object.load_model())
    ])


def tune_train()-> None:
    config_object=Configurations()
    config=config_object.training_configs()

    X_train, X_val, y_train, y_val = dataset.train_val_data(config=config['data'])
    (logger.info if logger else print)("Data Splitted for Train and Validation")

    results=[]
    for model_name in config["model"]["type"]:
        
        pipeline=pipeline_builder(model_name)

        pipeline_params = pipeline.get_params().keys()

        for p in config[model_name]['params']:
            if p not in pipeline_params:
                raise ValueError(f"Invalid pipeline parameter: {p}")
        
        train_object=Train(pipeline,model_name,
                           config[model_name]['params'],config['scoring'],config['target_matrix'])
        (logger.info if logger else print)("training Pipeline Build")

        tune_results=train_object.tune(X_train, y_train, X_val, y_val)

        results.append(tune_results)
        
    
    # Select best by primary metric
    results.sort(key=lambda r: r["best_score"], reverse=True)
    production = results[0]

    best_model_path = Path(config["model_dir"], "models", f"{production['model_name']}_best.joblib")

    best_model = load(best_model_path)
    (logger.info if logger else print)("Best Model Loaded From the given Path")
    
    best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    (logger.info if logger else print)("Refit on all Data")

    # Save final production model
    dump(best_model, Path(config["model_dir"], "models", "production_model.joblib"))
    Path(config["model_dir"], "results", "winner.json").write_text(json.dumps(production, indent=2))
    (logger.info if logger else print)("Production Model Saved")
        
if __name__=="__main__":
    tune_train()