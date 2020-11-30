import argparse
import glob
import os
import numpy as np
import sklearn
import torch
import librosa

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def data_list(sample_path):
    sub_dirs = [x[0] for x in os.walk(sample_path)]
    sub_dirs.pop(0)

    data_list = []

    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(sample_path, dir_name, '*')
        file_list.extend(glob.glob(file_glob))

        for file_name in file_list:
            data_list.append([file_name, dir_name])

    return np.array(data_list)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        wav, sr = librosa.load(self.data_list[idx][0], sr=None)
        label = int(self.data_list[idx][1])
        return wav, sr, label


def feature_extraction(wav, sr):
    features = []
    for idx, w in enumerate(wav):
        # mfcc = librosa.feature.mfcc(y=w, sr=sr[idx])
        # chromagram = librosa.feature.chroma_cqt(y=w, sr=sr[idx])
        features.append(w)

    return np.array(features)


def main():
    parser = argparse.ArgumentParser(description='Audio Classification')    
    parser.add_argument('-p', '--path', metavar='PATH', help='Path to Audio')
    args = parser.parse_args()

    DATA_PATH = 'D:/Download/Audio/'
    filelist = data_list(DATA_PATH)
    dataset = Dataset(filelist)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for data, rate, label in dataloader:
        x = data.numpy()
        y = label.numpy()
        sr = rate.numpy()

    features = feature_extraction(x, sr)

    # 10-fold cross validation
    acc_sum = 0
    cv = y.size
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        x_train = features[train_idx]
        x_test = features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Norm
        scaler = StandardScaler()
        scaler.fit(x_train)
        fscale_train = scaler.transform(x_train)
        fscale_test = scaler.transform(x_test)

        # SVC
        svc = SVC(kernel='rbf', random_state=1, gamma=0.001, C=1)
        model = svc.fit(fscale_train, y_train)

        predict_train = model.predict(fscale_train)
        correct_train = np.sum(predict_train == y_train)
        accuracy_train = correct_train / train_idx.size

        predict = model.predict(fscale_test)
        correct = np.sum(predict == y_test)
        accuracy = correct / test_idx.size
        print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        acc_sum += accuracy
        print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))
        print()


if __name__ == "__main__":
    main()