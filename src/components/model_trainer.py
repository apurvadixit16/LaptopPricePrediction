import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(alpha=0.001),
            "Ridge": Ridge(alpha=10),
            "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=3),
            "Decision Tree": DecisionTreeRegressor(max_depth=8),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100,
                                    random_state=3,
                                    max_samples=0.5,
                                    max_features=0.75,
                                    max_depth=15),
            "XGBRegressor": XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5), 
            "AdaBoost Regressor": AdaBoostRegressor(n_estimators=15,learning_rate=1.0)
            }

            model_report = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            #to get best model score from dict
            best_model_score = min(sorted(model_report.values()))

            #to get best model name from dict
            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
           

            if best_model_score <0.5:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)