from pydantic import BaseModel, Field, model_validator
from pathlib import Path
from datetime import datetime
from typing import Optional

class DataConfig:
    path: Path
    name: str
    seed: int
    test_size: float
    target: str


class APIData(BaseModel):
    VendorID: int = Field(..., gt=0, description="Vendorid")
    passenger_count: int = Field(..., gt=0, description="Number of passengers")
    RatecodeID: int = Field(..., gt=0, description="the rate code id")
    PULocationID: int = Field(..., gt=0, description="pickup location id")
    DOLocationID: int = Field(..., gt=0, description="drop off location id")
    payment_type: int = Field(..., gt=0, description="payment type 1 for credit, 2 cash")
    fare_amount: float = Field(..., gt=0, description="Total amount of the bill")
    improvement_surcharge: float = Field(..., gt=0, description="service improvement charges")
    trip_date: datetime = Field(..., description="Trip date in DD-MM-YYYY HH:MM:SS format", exclude=True)
    # derived fields
    day: Optional[str] = Field(default=None)
    hours: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def compute_day_features(self):
        try:
            self.day = self.trip_date.strftime('%A').lower()
            hours = self.trip_date.hour
            if 6<= hours <10:
                self.hours="am rush"
            elif 10<= hours <16:
                self.hours= "daytime"
            elif 16<= hours <20:
                self.hours="pm rush"
            else:
                self.hours="nighttime"
            return self
        except ValueError:
                raise ValueError("trip_date must be in YYYY-MM-DD HH:MM:SS format")
        
        

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "VendorID": 5,
                "passenger_count": 2,
                "RatecodeID": 1,
                "PULocationID": 100,
                "DOLocationID":231,
                "payment_type": 1,
                "fare_amount": 13,
                "improvement_surcharge": 0.3,
                "day": "monday",
                "hours": "am rush"
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    processed_at: str
    latency_ms: float