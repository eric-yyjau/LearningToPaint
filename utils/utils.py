
import datetime

# from utils.utils import getWriterPath
def getWriterPath(task='train', exper_name='', date=True):
    import datetime
    prefix = 'runs/'
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return prefix + task + '/' + exper_name + str_date_time

import numpy as np
# from utils.utils import crop_or_pad_choice
def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    # Adapted from https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93/fpointnet/data/datasets/utils.py
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points
    """
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, 'out_num_points = %d must be positive int!'%out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)
        choice = np.concatenate([choice, pad])
    return choice
