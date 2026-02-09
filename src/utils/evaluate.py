import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

class EvaluateMetrices:
    def __init__(self):
        pass

    def make_results(model_name:str, model_object, metric:str):

        # Create dictionary that maps input metric to actual metric name in GridSearchCV
        metric_dict = {'precision': 'mean_test_precision',
                    'recall': 'mean_test_recall',
                    'f1': 'mean_test_f1',
                    'accuracy': 'mean_test_accuracy',
                    }

        # Get all the results from the CV and put them in a df
        cv_results = pd.DataFrame(model_object.cv_results_)

        # Isolate the row of the df with the max(metric) score
        best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

        # Extract Accuracy, precision, recall, and f1 score from that row
        f1 = best_estimator_results.mean_test_f1
        recall = best_estimator_results.mean_test_recall
        precision = best_estimator_results.mean_test_precision
        accuracy = best_estimator_results.mean_test_accuracy

        # Create table of results
        table = pd.DataFrame({'model': [model_name],
                            'precision': [precision],
                            'recall': [recall],
                            'f1': [f1],
                            'accuracy': [accuracy],
                            },
                        )
        
        return table
    

    def get_test_scores(model_name:str, preds, y_test_data):
        '''
        Generate a table of test scores.

        In:
        model_name (string): Your choice: how the model will be named in the output table
        preds: numpy array of test predictions
        y_test_data: numpy array of y_test data

        Out:
        table: a pandas df of precision, recall, f1, and accuracy scores for your model
        '''
        accuracy = accuracy_score(y_test_data, preds)
        precision = precision_score(y_test_data, preds)
        recall = recall_score(y_test_data, preds)
        f1 = f1_score(y_test_data, preds)

        table = pd.DataFrame({'model': [model_name],
                            'precision': [precision],
                            'recall': [recall],
                            'f1': [f1],
                            'accuracy': [accuracy]
                            })

        return table