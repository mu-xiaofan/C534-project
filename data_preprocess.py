
import os
import numpy as np
import torch
import pickle as cp
from pandas import Series
from torch.utils.data import Dataset, DataLoader
from sliding_window import sliding_window


NUM_FEATURES = 9


def opp_sliding_window(data_x, data_y, window_size, step_size):
    """
    Create series data.
    """
    data_x = sliding_window(data_x, (window_size, data_x.shape[1]), (step_size, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, window_size, step_size)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def get_sample_weights(y, weights):
    """
    Assign weights to each sample.
    """
    label_unique = np.unique(y)
    sample_weights = []
    for val in y:
        idx = np.where(label_unique == val)
        sample_weights.append(weights[idx])
    return sample_weights


class data_loader(Dataset):
    # load data
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)

def load_data_dg():
    data_dir = './data/'
    saved_filename = 'dg_processed.data'
    if os.path.isfile( data_dir + saved_filename ) == True:
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X_train = data[0][0]
        y_train = data[0][1]
        X_val = data[1][0]
        y_val = data[1][1]
        X_test = data[2][0]
        y_test = data[2][1]
    else:
        str_folder = './data/dataset_fog_release/dataset/'
        DG_DATA_FILES_TRAIN = [
            'S01R01.txt',
            'S01R02.txt',
            'S03R01.txt',
            'S03R02.txt',
            'S03R03.txt',

            'S04R01.txt',

            'S05R01.txt',
            'S05R02.txt',

            'S06R01.txt',
            'S06R02.txt',

            'S07R01.txt',
            'S07R02.txt',

            'S08R01.txt',

            'S10R01.txt'
        ]
        DG_DATA_FILES_VAL = [
            'S09R01.txt'
        ]
        DG_DATA_FILES_TEST = [
            'S02R01.txt',
            'S02R02.txt'
        ]

        label = "2-class"
        print('\nProcessing train dataset files...\n')
        DG_DATA_FILES_TRAIN = [str_folder + a for a in DG_DATA_FILES_TRAIN]

        DG_DATA_FILES_VAL = [str_folder + a for a in DG_DATA_FILES_VAL]
        DG_DATA_FILES_TEST = [str_folder + a for a in DG_DATA_FILES_TEST]
        X_train, y_train = load_data_files(label, DG_DATA_FILES_TRAIN)
        print('\nProcessing VAL dataset files...\n')
        X_val, y_val = load_data_files(label, DG_DATA_FILES_VAL)
        print('\nProcessing test dataset files...\n')
        X_test, y_test = load_data_files(label, DG_DATA_FILES_TEST)
        print("Final datasets with size: | train {0} | val {1} | test {2}".format(X_train.shape, X_val.shape, X_test.shape))

        obj = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_data_files(label, data_files):
    """
    Get feature x and label y
    """
    data_x = np.empty((0, NUM_FEATURES))
    data_y = np.empty((0))

    for filename in data_files:
        try:
            # data = np.loadtxt(BytesIO(zipped_dataset.read(filename)))
            data = np.loadtxt(filename)
            print('... file {0}'.format(filename))
            x, y = process_dataset_file(data, label)
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
            # print("Data's shape yet: ", data_x.shape())
        except KeyError:
            print('ERROR: Did not find {0} in zip file'.format(filename))
    return data_x, data_y


def process_dataset_file(data, label):
    data = select_row_col_dg(data, label)

    data_x, data_y = divide_x_y(data, label)

    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Filling in NaN
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0
    data_x = normalize(data_x)
    return data_x, data_y


def select_row_col_dg(data, label):
    """
    :param data:
    :param label:
    :return: select the desired data (binary VS multi-class)
    """
    if label == "2-class":
        zero_ind = [i for i, e in enumerate(data[:, -1]) if e == 0]
        data = np.delete(data, zero_ind, 0)

        col_to_delete = 0
        data = np.delete(data, col_to_delete, 1)
        return data


def divide_x_y(data, label):
    """Segments each sample into (time+features) and (label)

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """
    # the last column is label vector
    data_y = data[:, -1]
    # remove the last column
    data_x = np.delete(data, 9, 1)
    if label not in ['2-class', '3-class']:
            raise RuntimeError("Invalid label: '%s'" % label)
    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """

    if label == '2-class':  # Labels for locomotion are adjusted
        data_y[data_y == 1] = 0
        data_y[data_y == 2] = 1
    elif label == '3-class':  # Labels for gestures are adjusted
        pass
    return data_y


def normalize(x):
    """Normalizes all sensor channels by mean substraction,
    dividing by the standard deviation and by 2.

    :param x: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    x = np.array(x, dtype=np.float32)
    m = np.mean(x, axis=0)
    x -= m
    std = np.std(x, axis=0)
    std += 0.000001

    x /= std
    return x

def transform_features_for_ml(x_windows):
    # convert every window's feature into mean and std to fit ml input
    transformed = np.zeros((x_windows.shape[0], 18))  # every window 18 features（9 mean + 9 std）

    for i in range(x_windows.shape[0]):
        feature_means = np.mean(x_windows[i], axis=0)
        feature_stds = np.std(x_windows[i], axis=0)
        transformed[i] = np.concatenate((feature_means, feature_stds))
    return transformed


def load_dataset_dl(batch_size=64, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_dg()
    x_train_win, y_train_win = opp_sliding_window(x_train, y_train, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    x_val_win, y_val_win = opp_sliding_window(x_val, y_val, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    x_test_win, y_test_win = opp_sliding_window(x_test, y_test, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)

    unique_ytrain, counts_ytrain = np.unique(y_train_win, return_counts=True)

    weights = 100.0 / torch.Tensor(counts_ytrain)
    weights = weights.double()
    sample_weights = get_sample_weights(y_train_win, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_set = data_loader(x_train_win, y_train_win)
    val_set = data_loader(x_val_win, y_val_win)
    test_set = data_loader(x_test_win, y_test_win)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print('train_loader batch: ', len(train_loader), 'test_loader batch: ', len(test_loader))
    return train_loader, val_loader, test_loader


def load_dataset_ml(SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_dg()
    x_train_win, y_train_win = opp_sliding_window(x_train, y_train, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    x_val_win, y_val_win = opp_sliding_window(x_val, y_val, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    x_test_win, y_test_win = opp_sliding_window(x_test, y_test, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    x_train_transformed = transform_features_for_ml(x_train_win)
    x_val_transformed = transform_features_for_ml(x_val_win)
    x_test_transformed = transform_features_for_ml(x_test_win)
    return x_train_transformed, x_val_transformed, x_test_transformed, y_train_win, y_val_win, y_test_win