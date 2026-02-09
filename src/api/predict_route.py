from fastapi import APIRouter, HTTPException
import time
from datetime import datetime
from src.utils.schemas import APIData, PredictionResponse
from src.api.wrapper import Wrapper

router=APIRouter(tags=['prediction'])

@router.post("/predict",response_model=PredictionResponse,status_code=200)
async def prediction(data: APIData):

    """
    Asynchronous endpoint for high-performance ML inference.
    """
    W_object= Wrapper()
    start_time = time.perf_counter()
    
    try:
        # Step 1: model loading
        #W_object.load_model()
        
        # Step 2: Inference
        result = W_object.prediction_engine(data)
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return PredictionResponse(
            prediction= result,#['generous' if value==1 else 'not generous' for value in result],
            processed_at=datetime.utcnow().isoformat(),
            latency_ms=round(latency, 2)
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        #logging.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference failure")        