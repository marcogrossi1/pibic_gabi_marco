import language

if language.language == 'pt':
    import messages.messagesPT as msg

else:
    import messages.messagesEN as msg

import functionalities.mnistDATA as data
import functionalities.mnistPLOT as plot

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import time

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def shortAnn(activation, solver, nHiddenLayers):
    start = time.perf_counter()

    print(msg.CREATING)
        
    xTrain, xTest, yTrain, yTest = data.getXY(0.15, 1)
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)
    
    layers = list(map(int, nHiddenLayers.split()))

    model = MLPClassifier(hidden_layer_sizes=layers, solver=solver, activation=activation, max_iter=50, verbose=False, random_state=1)
    
    print(msg.CREATED)
    print(msg.SHORT_TRAINING)
    
    model.fit(xTrain, yTrain)

    print(msg.DONE)
    print(msg.TRAINING_SCORE, model.score(xTrain, yTrain))
    print(msg.TESTING_SCORE, model.score(xTest, yTest))

    index = 0
    test_digit = xTest[index].reshape(1, -1)
    preds = model.predict(xTest)
    
    print(msg.DIGITS_PLOT)
    print(msg.CLOSE_IMG)
    plot.plotDigits(xTest=xTest, model=model, yTest=yTest)

    print(msg.CONFUSION_PLOT)
    print(msg.CLOSE_IMG)
    plot.plotConfusion(yTest=yTest, preds=preds)

    end = time.perf_counter()
    print(f"Elapsed Time: {end - start:.2f} seconds")

def longANN():
    start = time.perf_counter()

    print(msg.CREATING)

    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs', 'sgd', 'adam']
    hiddenLayers = [[10], [50], [100], [100, 50]]

    xTrain, xTest, yTrain, yTest = data.getXY(0.15, 1)
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)

    bestScore = 0
    bestPred = 0

    results = []

    print(msg.CREATED)
    print(msg.LONG_TRAINING)
    j = 1
    for solver in solvers:
        for hiddenLayer in hiddenLayers:
            for activation in activations:
                model = MLPClassifier(hidden_layer_sizes=hiddenLayer, solver=solver, activation=activation, max_iter=100, verbose=False, random_state=1)
                model.fit(xTrain, yTrain)

                score = model.score(xTest, yTest)

                print(msg.LINE)
                print(msg.ACTIVE_TESTING(j=j, solver=solver, hiddenLayer=hiddenLayer, activation=activation))

                results.append((model, solver, hiddenLayer, score))

                index = 0
                test_digit = xTest[index].reshape(1, -1)

                preds = model.predict(xTest)

                if score > bestScore:
                    bestPred = preds

                j += 1

    plot.plotConfusion(yTest=yTest, preds=bestPred)

    print(msg.FINAL_RESULTS)
    print(results)

    end = time.perf_counter()
    print(f"Elapsed Time: {end - start:.2f} seconds")