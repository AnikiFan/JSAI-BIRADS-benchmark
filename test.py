import random
import unittest
import pandas as pd
import numpy as np
import os
from utils import TableDataset

class ClaTestCase(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.test_images = os.listdir(os.path.join(os.curdir,'test'))
        self.num_classes = 6
        self.seed = 42
        self.labels = np.random.randint(low=0,high=self.num_classes,size =len(self.test_images))
        self.pesu_table = pd.DataFrame({
            "file_name" : self.test_images,
            "label": self.labels
        })

    def test_TableDataset(self):
        tableDataset = TableDataset(self.pesu_table)

if __name__ == '__main__':
    unittest.main()
