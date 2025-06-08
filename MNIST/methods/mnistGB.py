import language

if language.language == 'pt':
    import messages.messagesPT as msg

else:
    import messages.messagesEN as msg

import functionalities.mnistDATA as data
import functionalities.mnistPLOT as plot

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import time

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def shortGB():
    start = time.perf_counter()

    print(msg.CREATING)

    estimators = [100, 200, 500, 1000]
    learningRates = [1.0, 0.5, 2.0]
    maxDepths = [1, 3, 5, 10]

    xTrain, xTest, yTrain, yTest = data.getXY(0.15, 1)
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, verbose=1, random_state=1)
    
    print(msg.CREATED)
    print(msg.LONG_TRAINING)

    model.fit(xTrain, yTrain)

    print(msg.DONE)
    print(msg.TRAINING_SCORE, model.score(xTrain, yTrain))
    print(msg.TESTING_SCORE, model.score(xTest, yTest))

    index = 0
    test_digit = xTest[index].reshape(1, -1)
    preds = model.predict(xTest)

    plot.plotConfusion(yTest=yTest, preds=preds)

    end = time.perf_counter()
    print(f"Elapsed Time: {end - start:.2f} seconds")