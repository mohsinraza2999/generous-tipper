import yaml
from typing import Dict
from pathlib import Path
import os
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

class Configurations:
    def __init__(self):
        CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/app/config"))
        self.TRAIN_CFG = CONFIG_DIR / "train_config.yml"
        self.DATA_CFG=CONFIG_DIR / "data_config.yml"
        
    def training_configs(self)-> Dict[str,str]:
        
        with open(self.TRAIN_CFG, "r") as f:
            
            return self._validate_param_grid(yaml.safe_load(f))
        

    def data_configs(self)-> Dict[str,str]:
        with open(self.DATA_CFG, "r") as f:

            return yaml.safe_load(f)


    def _validate_scoring(self,scoring_cfg)-> None:
        
        mapping = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="binary"),
            "recall": make_scorer(recall_score, average="binary"),
            "f1": make_scorer(f1_score, average="binary"),
        }
        for key, name in scoring_cfg.items():
            if name not in mapping:
                raise ValueError(f"Unknown scorer '{name}'. Allowed: {list(mapping.keys())}")
            

    def _validate_param_grid(self,configurations) -> None:

        self._validate_scoring(configurations['scoring'])

        for model_name in configurations["model"]["type"]:

            if not configurations[model_name]['params']:
                raise ValueError(f"No params provided for '{model_name}'.")
            

        return configurations
            # Ensure pipeline prefix
            #if not model_name.startswith("model__"):
            #    raise ValueError(
            #        f"Param '{model_name}' must be prefixed with 'model__' for Pipeline. Model: {model_name}"
            #    )