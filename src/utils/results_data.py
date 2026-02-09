import pandas as pd
from pathlib import Path
from joblib import dump
import json

def save_results(new_row:pd.DataFrame,path:str='artifacts'):

    if Path(path,"results/results.parquet").exists():
        df = pd.read_parquet(path)
        df = pd.concat([df, new_row], axis=0,ignore_index=True)
    else:
        df = new_row
        path/"results".mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(path,"results/results.parquet"), index=False)

def save_best(best_model, model_name:str, best_params, best_score, path:str='artifacts')->None:

    Path(path, "results").mkdir(parents=True, exist_ok=True)
    Path(path, "models").mkdir(parents=True, exist_ok=True)

    Path(path, "results", f"{model_name}_best.json").write_text(json.dumps({
        "best_parameters": best_params, "best_score": best_score}, indent=2))

    # Save refit model
    dump(best_model, Path(path, "models", f"{model_name}_best.joblib"))