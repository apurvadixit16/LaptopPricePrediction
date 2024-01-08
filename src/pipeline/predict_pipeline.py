import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        company : str,
        laptop_type : str,
        ram :int,
        weight : int,
        touchscreen : int,
        ips : int,
        screen_size : int,
        resolution :int,
        cpu ,
        hdd ,
        ssd ,
        gpu ,
        os,
        ppi):

        self.company = company
        self.laptop_type = laptop_type
        self.ram = ram
        self.weight = weight
        self.touchscreen = touchscreen
        self.ips = ips
        self.screen_size = screen_size
        self.resolution = resolution
        self.cpu = cpu
        self.hdd = hdd
        self.cpu = cpu
        self.ssd = ssd
        self.gpu = gpu
        self.os = os
        self.ppi = ppi

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Company": [self.company],
                "TypeName": [self.laptop_type],
                "Ram": [self.ram],
                "Weight": [self.weight],
                "touchscreen": [self.touchscreen],
                "ips": [self.ips],
                "ppi": [self.ppi],
                "screen_size": [self.screen_size],
                "resolution": [self.resolution],
                "Cpu brand": [self.cpu],
                "HDD": [self.hdd],
                "SSD": [self.ssd],
                "Gpu brand": [self.gpu],
                "os": [self.os]
            }
            print(pd.DataFrame(custom_data_input_dict))
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)