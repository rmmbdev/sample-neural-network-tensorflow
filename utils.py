import numpy as np


def create_normal_file():
    with open('iris.data') as f:
        data = f.read()

    data_lines = data.split('\n')

    data = []
    for line in data_lines:
        data.append(line.split(','))

    # remove last empty cells
    data_str = data[0:-2]

    data = np.asarray(data_str)

    data[0:50, -1] = '1'
    data[50:100, -1] = '2'
    data[100:, -1] = '3'

    data = np.asarray(data, dtype=np.float)

    np.save('iris-data', data)


def noramlize(datax, minx, maxx):
    return (2 * ((datax - minx) / (maxx - minx))) - 1


def noramlize01(datax, minx, maxx):
    return (datax - minx) / (maxx - minx)


def to_onehot(labels, class_count):
    outlabels = np.zeros((len(labels), class_count))
    for i, l in enumerate(labels):
        l = int(l)
        outlabels[i, l] = 1
    return outlabels


if __name__ == "__main__":
    create_normal_file()
