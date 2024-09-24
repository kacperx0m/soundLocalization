from sklearn import linear_model
from sklearn.model_selection import train_test_split
from feature_extraction import draw_ITD_chart
from pathlib import Path
import numpy as np
from pickle import dump, load

def save_model(model, path):
    """
    path requires .pkl extension
    """
    with open(path, "wb") as f:
        dump(model, f, protocol=5)

def load_model(path):
    """
    path requires .pkl extension
    """
    with open(path, "rb") as f:
        model = load(f)
    return model

def load_data_from_dir(directory, ret_earlier=False):
    directory = sorted(directory, key=lambda x: int(x.stem))
    ITD, ILD = draw_ITD_chart(directory)
    X, Y = np.array([ITD, ILD]).T, [float(file.stem) for file in directory]
    if ret_earlier:
        return X, Y
    # print(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

def train_model_on_databases(*databases):
    reg = linear_model.LinearRegression()
    for arg in databases:
        directory = list(Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/"+arg).iterdir())
        x_train, x_test, y_train, y_test = load_data_from_dir(directory)
        reg.fit(x_train, y_train)
        print(reg.score(x_test, y_test))

    return reg

# directory = list(Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/frogsDB/").iterdir())
# directory = sorted(directory, key=lambda x: int(x.stem))
# print(directory)
# ITD, ILD = draw_ITD_chart(directory)
# print(ITD.__len__(), ILD.__len__())

'''
X, Y = np.array([ITD, ILD]).T, [file.stem for file in directory]
# print(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
pred = reg.predict(x_test)
print(pred, y_test)

directory = list(Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/horseDB/").iterdir())
directory = sorted(directory, key=lambda x: int(x.stem))
# print(directory)
ITD, ILD = draw_ITD_chart(directory)

X, Y = np.array([ITD, ILD]).T, [file.stem for file in directory]
# print(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

pred = reg.predict(x_test)
print(pred, y_test)
'''

# model = train_model_on_databases("frogsDB/", "horseDB/", "musicDB/")
# model = load_model("C:/Users/uzytek/PycharmProjects/inzynierka/models/model1.pkl")
# save_model(model, "C:/Users/uzytek/PycharmProjects/inzynierka/models/model1.pkl")

directory = list(Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/musicDB/").iterdir())

reg = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = load_data_from_dir(directory)
reg.fit(x_train, y_train)
pred = reg.predict(x_test)
print(np.round(pred,1), y_test)