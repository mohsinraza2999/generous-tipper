from pathlib import Path
from joblib import load
import pandas as pd
from sklearn.pipeline import Pipeline
from src.utils.schemas import APIData
from src.utils.configs import Configurations

class Wrapper:

    def __init__(self, model_path: str = None):
        config=Configurations().training_configs()
        self.model_dir = Path(config["model_dir"], "models")
        self.model_path = Path(model_path) if model_path else self.model_dir / "production_model.joblib"
        self.model: Pipeline | None = None
    # --- 2. LOGIC WRAPPERS ---

    def preprocess_input(self, data: APIData) -> pd.DataFrame:
        """Convert API input to DataFrame suitable for the pipeline."""
        # Convert Pydantic / dict to DataFrame
        try:
            return pd.DataFrame(data.model_dump(),index=[0])
        except:
            raise ValueError("cannot be convert into dataframe")
    

    def load_model(self) -> None:

        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        #Professional step with error handling.
        try:
            #Load trained pipeline from disk.
            self.model = load(self.model_path)
        except Exception as e:
            raise ValueError(f"Model loading failed: {str(e)}")

    def prediction_engine(self, data: APIData) -> str:
        #Run inference using the loaded pipeline.

        #if self.model is None:
        #    raise RuntimeError("Model not loaded. Call load_model() first.")

        X = self.preprocess_input(data)
        # Simulation:predicting if they are a generous tipper
        #is_generous = self.model.predict(X)
        print ("\n X class =", X.__class__)
        print("\n X \n", X)
        return str(data.model_dump()) #is_generous.tolist()