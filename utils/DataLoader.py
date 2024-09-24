import os

class CrossValidationDataLoader:
    official_data_path = os.path.join(os.pardir,'data','breast','myTrain','cla')
    BUS_data_path = os.path.join(os.pardir,'data','breast','BUS','Images')
    USG_data_path = os.path.join(os.pardir,'data','breast','USG')
    def __init__(self,k_fold=5,BUS=True,USG=True):