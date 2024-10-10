from utils.OfficialClaDataOrganizer import OfficialClaDataOrganizer
from utils.OfficialFeaDataOrganizer import OfficialFeaDataOrganizer
import os
if __name__ == '__main__':
    # cla数据组织
    pre_dir = os.path.join(os.getcwd(), 'data', 'breast', 'cla')
    src = os.path.join(pre_dir, 'official_test')
    dst = os.path.join(pre_dir, 'test')
    OfficialClaDataOrganizer(src,dst).organize(ignore=True)
    src = os.path.join(pre_dir, 'official_train')
    dst = os.path.join(pre_dir, 'train')
    OfficialClaDataOrganizer(src,dst).organize(ignore=False)
    # fea数据组织
    pre_dir = os.path.join(os.getcwd(), 'data', 'breast', 'fea')
    src = os.path.join(pre_dir, 'official_test')
    dst = os.path.join(pre_dir, 'test')
    OfficialFeaDataOrganizer(src,dst).organize(ignore=True)
    src = os.path.join(pre_dir, 'official_train')
    dst = os.path.join(pre_dir, 'train')
    OfficialFeaDataOrganizer(src,dst).organize(ignore=False)
