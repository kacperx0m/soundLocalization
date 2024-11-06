import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from feature_extraction import calculate_features_on_directory
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
    # print(list(directory))
    directory = sorted(directory, key=lambda x: int(x.stem.replace(x.parent.name,"")))
    # print(list(directory))
    ITD, ILD = calculate_features_on_directory(directory, plot=False)
    X, Y = np.array([ITD, ILD]).T, range(-90,91,1) #[float(file.stem.replace(file.parent.name,"")) for file in directory]
    if ret_earlier:
        return X, Y
    # print(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

def train_model_on_databases(*databases, model):
    # reg = LinearRegression()
    # reg = DecisionTreeRegressor()  # try max_depth from 8 to 2
    # reg = svm.SVR()
    # reg = KNeighborsRegressor()
    X_all, Y_all = [], []
    for arg in databases:
        print(f"now iterating over {arg}")
        # directory = list(Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/"+arg).iterdir())
        # directory = list(Path(arg).iterdir())
        directory = Path(arg).iterdir()
        X, Y = load_data_from_dir(directory, ret_earlier=True)
        X_all.append(X)
        Y_all.append(Y)
        # print("score: ",reg.score(x_test, y_test))
        # return 0
    X_all = np.vstack(X_all)
    Y_all = np.concatenate(Y_all)
    # print("training: ", X_all, Y_all)

    x_train, x_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=0)
    print("training started. ")
    model.fit(x_train, y_train)
    print("score: ", model.score(x_test, y_test))

    return model

# directory = list(Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/frogsDB/").iterdir())
# directory = sorted(directory, key=lambda x: int(x.stem))
# print(directory)
# ITD, ILD = calculate_features_on_directory(directory)
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
ITD, ILD = calculate_features_on_directory(directory)

X, Y = np.array([ITD, ILD]).T, [file.stem for file in directory]
# print(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

pred = reg.predict(x_test)
print(pred, y_test)
'''

# model = train_model_on_databases("frogsDB/", "horseDB/", "musicDB/")
# directory = list(Path("D:/inzynierka_baza/wynikowy/").iterdir())
directory = list(Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/lufs_horseDB").iterdir())  #.glob("1*.wav"))
# print(directory)
# print(*directory[:1])
# model = train_model_on_databases(*directory[:1])
# linear_regressor = LinearRegression()
# tree_regressor = DecisionTreeRegressor() # try max_depth from 8 to 2
# forest_regressor = RandomForestRegressor()
# svm_regressor = svm.SVR()
# knn_regressor = KNeighborsRegressor()
# model = train_model_on_databases(*directory, model=forest_regressor)
regression_model = load_model("D:/inzynierka_baza/model_caly_ds_linear.pkl")
# save_model(model, "D:/inzynierka_baza/model_caly_ds_forest.pkl")
# model = load_model("D:/inzynierka_baza/model_testowy_caly_ds_forest.pkl")

# directory = list(Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/musicDB/").iterdir())
#
# reg = linear_model.LinearRegression()
# reg.fit(x_train, y_train)

def test_saved_model(directory, model):
    x_train, x_test, y_train, y_test = load_data_from_dir(directory)
    pred = model.predict(x_test)
    # print(np.clip(pred, -90, 90).round(pred), np.array(y_test))
    print(np.array(pred).round(), np.array(y_test))
    return np.array(pred).round(), np.array(y_test)

pred, real = test_saved_model(directory, regression_model)
plt.plot(pred)
plt.plot(real)
plt.show()
print(max(abs(np.abs(real) - np.abs(pred))))
print(regression_model.get_params())
#
# model = load_model("D:/inzynierka_baza/model_testowy_caly_ds_tree.pkl")
# pred, real = test_saved_model(directory, model)
# plt.plot(pred)
# plt.plot(real)
# plt.show()
#
# model = load_model("D:/inzynierka_baza/model_testowy_caly_ds_svr.pkl")
# pred, real = test_saved_model(directory, model)
# plt.plot(pred)
# plt.plot(real)
# plt.show()
#
# model = load_model("D:/inzynierka_baza/model_testowy_caly_ds_knn.pkl")
# pred, real = test_saved_model(directory, model)
# plt.plot(pred)
# plt.plot(real)
# plt.show()

def test():
    directory = list(Path("D:/inzynierka_baza/wynikowy-test/01_Kick/").iterdir())
    # directory = list(Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/frogsDB/").iterdir())
    directory = sorted(directory, key=lambda x: int(x.stem.replace(x.parent.name,"")))
    regression_model = LinearRegression()
    x_train, x_test, y_train, y_test = load_data_from_dir(directory)
    regression_model.fit(x_train, y_train)

    directory = list(Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/lufs_musicDB").iterdir())
    x_train, x_test, y_train, y_test = load_data_from_dir(directory)
    pred = regression_model.predict(x_test)
    print(np.round(pred,1), np.array(y_test))

# def calculate_features_on_directory(directory):
#     X, Y = load_data_from_dir(directory, ret_earlier=True)
#     plt.plot(X,Y)
#     plt.show()

# calculate_features_on_directory(directory, plot=True)

