import pandas as pd
import os


def getClassDistribution(dataset_path:str = os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI'))->dict:
    table = pd.read_csv(os.path.join(dataset_path, 'ground_truth.csv'))
    # print(table.head())
    class_distribution = table['label'].value_counts().to_dict()
    class_distribution_list = [class_distribution[i] for i in range(6)]
    return class_distribution, class_distribution_list


if __name__ == '__main__':
    class_distribution, class_distribution_list = getClassDistribution()
    print(class_distribution)
    print(class_distribution_list)