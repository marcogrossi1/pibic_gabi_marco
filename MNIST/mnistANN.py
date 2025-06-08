import mnistDATA as data
import mnistPLOT as plot

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import time

def shortAnn(activation, solver, nHiddenLayers):
    start = time.perf_counter()

    print("Criando o modelo...\n")
        
    xTrain, xTest, yTrain, yTest = data.getXY(0.15, 1)
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)
    
    layers = list(map(int, nHiddenLayers.split()))

    model = MLPClassifier(hidden_layer_sizes=layers, solver=solver, activation=activation, max_iter=50, verbose=1, random_state=1)
    
    print("Modelo criado!\n")
    print("Treinando...\n")
    
    model.fit(xTrain, yTrain)

    print("Pronto!\n")
    print("Training set score:", model.score(xTrain, yTrain))
    print("Testing set score:", model.score(xTest, yTest))  

    index = 0
    test_digit = xTest[index].reshape(1, -1)
    preds = model.predict(xTest)
    
    print("Aqui vão alguns exemplos de dígitos!\n")
    print("Feche a imagem para prosseguir.\n")
    plot.plotDigits(xTest=xTest, model=model, yTest=yTest)

    print("Aqui vai o Confusion Plot!\n")
    print("Feche a imagem para prosseguir.\n")
    plot.plotConfusion(yTest=yTest, preds=preds)

    end = time.perf_counter()
    print(f"Tempo Total: {end - start:.2f} segundos")

def longANN():
    start = time.perf_counter()

    print("Criando o modelo...")

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

    print("Iniciando os testes. Se acomode porque isso vai demorar!")
    j = 0
    for solver in solvers:
        for hiddenLayer in hiddenLayers:
            for activation in activations:
                model = MLPClassifier(hidden_layer_sizes=hiddenLayer, solver=solver, activation=activation, max_iter=50, verbose=False, random_state=1)
                model.fit(xTrain, yTrain)

                score = model.score(xTest, yTest)

                print("---------------------------------------")
                print(f"Test {j}")
                print(f"Model: {model}, Solver: {solver}, Layers: {hiddenLayer}")
                print("Training set score:", model.score(xTrain, yTrain))
                print("Testing set score:", score)

                results.append((model, solver, hiddenLayer, score))

                index = 0
                test_digit = xTest[index].reshape(1, -1)

                preds = model.predict(xTest)

                if score > bestScore:
                    bestPred = preds

                j += 1

    plot.plotConfusion(yTest=yTest, preds=bestPred)

    print("Estes são os resultados de todos os testes:")
    print(results)

    end = time.perf_counter()
    print(f"Tempo Total: {end - start:.2f} segundos")