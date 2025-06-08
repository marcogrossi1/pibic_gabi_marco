from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)

data.head()

data = data / 255.0

def getXY(testSize, randomState):
    return train_test_split(data, labels, test_size=testSize, random_state=randomState)