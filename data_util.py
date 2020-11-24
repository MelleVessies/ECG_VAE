import os
import numpy as np
from tqdm import tqdm
import wfdb
import pandas as pd
import ast
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


def load_PTB_XL_meta(path: str):
    """
    Load PTB_XL meta data

    :param path: General PTB_XL folder path; Should contain ptbxl_database.csv as well as raw data
    :return: DataFrame with PTB_XL meta data
    """
    # load and convert annotation data
    meta = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    meta.scp_codes = meta.scp_codes.apply(lambda x: ast.literal_eval(x))

    return meta


def load_PTB_XL_to_memory(path: str, sampling_rate: int):
    """
    Returns ALL raw ECG data and metadata for specified sampling rate.

    :param path: General PTB_XL folder path; Should contain ptbxl_database.csv as well as raw data
    :param sampling_rate: Sampling rate of ECG data
    :return: X: Raw ECG data in numpy format; Y: ptbxl_database.csv as pandas dataframe;
    """
    # Load meta
    Y = load_PTB_XL_meta(path)

    # Load raw signal data
    X = load_raw_data_to_memory(Y, sampling_rate, path)

    return X, Y


def load_raw_data_to_memory(df: pd.DataFrame, sampling_rate: int, path: str):
    """
    Returns ALL raw ECG data for specified sampling rate.

    :param df: Metadata dataframe containing file names of raw data
    :param sampling_rate: Sampling rate of ECG
    :param path: General PTB_XL folder path; Should contain ptbxl_database.csv as well as raw data
    :return:
    """

    data = None
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            data.dump(path+'raw100.npy')
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            data.dump(path+'raw500.npy')

    # assert(data is not None, "None data found after data loading")
    return data


def dump_wfdb_to_numpy(df: pd.DataFrame, sampling_rate: int, path: str):
    """
    Converts and dumps each raw wfdb formatted ECG entry to numpy. Numpy data loads significantly faster.

    :param df: Metadata dataframe containing file names of raw data
    :param sampling_rate: Sampling rate of ECG
    :param path: General PTB_XL folder path; Should contain ptbxl_database.csv as well as raw data
    :return:
    """

    filenames = df.filename_lr if sampling_rate == 100 else df.filename_hr
    for fn in tqdm(filenames):
        np_fn = path + fn + ".npy"
        if not os.path.isfile(np_fn):
            signal, meta = wfdb.rdsamp(path+fn)
            signal = np.array(signal)
            signal.dump(np_fn)


class PTBXLDataset(Dataset):
    """
        Pytorch dataset class for the PTBXL dataset
    """
    def __init__(self, meta: pd.DataFrame, sampling_rate: int, path: str = './PTB_XL/', use_numpy: bool = True):
        self.meta = meta
        self.filenames = meta.filename_lr if sampling_rate == 100 else meta.filename_hr
        self.filenames = self.filenames.values.tolist()
        self.sampling_rate = sampling_rate
        self.path = path
        self.use_numpy = use_numpy

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        fn = self.filenames[index]

        if self.use_numpy:
            data = np.load(self.path + fn + ".npy", allow_pickle=True)
        else:
            signal, signal_meta = wfdb.rdsamp(self.path + fn)
            data = np.array(signal)

        data = torch.from_numpy(data.swapaxes(0, 1)).float()

        # for VAE input and target are the same
        return data, data


class PTBXLDataModule(pl.LightningDataModule):
    """
        Pytorch lightning data-module class for the PTBXL dataset, automatically creates train, dev/val and test sets
    """
    def __init__(self, data_dir: str = './PTB_XL/', sample_rate: int = 100, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.ptb_xl_meta = load_PTB_XL_meta(self.data_dir)

        self.train, self.dev, self.test = None, None, None

    def prepare_data(self):
        dump_wfdb_to_numpy(self.ptb_xl_meta, self.sample_rate, self.data_dir)

    def setup(self, stage=None):
        train_meta, test_meta = train_test_split(self.ptb_xl_meta, test_size=0.1)
        train_meta, dev_meta = train_test_split(train_meta, test_size=0.1)

        self.train = PTBXLDataset(train_meta, self.sample_rate, self.data_dir, True)
        self.dev = PTBXLDataset(dev_meta, self.sample_rate, self.data_dir, True)
        self.test = PTBXLDataset(test_meta, self.sample_rate, self.data_dir, True)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dev, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=2)


if __name__ == '__main__':
    dataMod = PTBXLDataModule()

