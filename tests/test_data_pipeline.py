import pandas as pd
from pathlib import Path
import pytest
from src.data_pipeline.dataset import ReadData, train_val_data
from src.utils.configs import Configurations
from src.training_pipeline.train import Train, pipeline_builder

@pytest.fixture
def data_config():
    config_object = Configurations()
    return config_object.data_configs()

@pytest.fixture
def train_config():
    config_object = Configurations()
    return config_object.training_configs()



def test_read_class(data_config)-> None:

    read_object= ReadData(Path(data_config['raw_path']))

    df=read_object.read(data_config['name'])

    assert isinstance(df, pd.DataFrame) #If df is a DataFrame, it cannot be None.

def test_train_val_split(train_config)-> None:

    X_train, y_train, X_test, y_test= train_val_data(train_config['data'])

    assert any(X_train) or any(y_train) or any(X_test) or any(y_test)




def test_pipeline_builds(train_config):
    for model_name in train_config['model']['type']:
        pipeline = pipeline_builder(
            model_type=model_name
        )
        assert pipeline is not None
    
def test_param_grid_matches_pipeline(train_config):
    for model_name in train_config['model']['type']:
        pipeline = pipeline_builder(
            model_type=model_name
        )
        train_object=Train(pipeline,model_name,
                       train_config[model_name]['params'],train_config['scoring'],
                       train_config['target_matrix'])

        pipeline_params = train_object.tune_tipper.get_params().keys()

        for param in train_config[model_name]['params']:
            assert param in pipeline_params, f"{param} not in pipeline"

def test_pipeline_predict_shape(train_config):
    for model_name in train_config['model']['type']:
        pipeline = pipeline_builder(
            model_type=model_name
        )						
        X = pd.DataFrame({
            "VendorID": [2, 1, 1, 2],
            "passenger_count": [6, 1, 1, 1],
            "RatecodeID": [1, 1, 1, 1],
            "PULocationID":[100,186,223,256],
            "DOLocationID":[231, 43, 236, 97],
            "payment_type":[1,1,1,1],
            "fare_amount":[13.0, 16.0 , 6.5, 20.5],
            "improvement_surcharge":[0.3, 0.3, 0.3, 0.3],
            "day":["saturday", "tuesday", "friday", "sunday"],
            "hours":["am rush", "daytime", "am rush", "daytime"]
        })

        y = [1, 1, 0, 1]

        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        assert len(preds) == len(X)
    

"""

    def test_valid_date_parsing():
        trip = TripDate(trip_date="2024-10-15")

        assert trip.date_obj == datetime(2024, 10, 15)
        assert trip.day == 15
        assert trip.month == 10
        assert trip.year == 2024
        assert trip.day_of_week == "Tuesday"
        assert trip.is_weekend is False

"""


"""#    import pickle


    def test_pipeline_serialization():
        pipeline = build_pipeline(
            num_features=["num1"],
            cat_features=["cat1"]
        )

        X = pd.DataFrame({
            "num1": [1, 2],
            "cat1": ["a", "b"]
        })
        y = [0, 1]

        pipeline.fit(X, y)

        dumped = pickle.dumps(pipeline)
        loaded = pickle.loads(dumped)

        preds = loaded.predict(X)

        assert len(preds) == len(X)
            

def main()-> None:
    test_object= TestDataLoader()

    print("test 1 read class")

    test_object.test_read_class()

    print("test 2 train val split")

    test_object.test_train_val_split()


if __name__=="__main__":
    main()"""