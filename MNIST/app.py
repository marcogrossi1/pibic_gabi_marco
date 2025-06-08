print("Instalando dependencias necessárias! Esse processo pode demorar alguns segundos.")

import importlib.metadata, subprocess, sys

required  = {'numpy','scikit-learn', 'pandas', 'seaborn', 'matplotlib'}
installed = {pkg.metadata['Name'] for pkg in importlib.metadata.distributions()}
missing   = required - installed

if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

print("Iniciando a aplicação...")

import mnistANN
import mnistGB

auxLine = "---------------------------------------"

print("Aplicação iniciada!")
print(auxLine)
print("Olá! Vamos resolver o problema dos dígitos do MNIST utilizando métodos de ML!!!")
print(auxLine)
print("Qual método você deseja utilizar?")
print("1. Artificial Neural Network - ANN")
print("2. Gradient Boosting - GB")

n = int(input())

# ANN
if n == 1:
    print(auxLine)
    print("Você prefere testar os parametros?")
    print("1. Sim, quero encontrar os melhores parametros dentre os disponíveis (tempo estimado 30 min)")
    print("2. Não, prefiro rodar com padrões loucos e fé (tempo estimado 45 seg)")

    n = int(input())

    if n == 1:
        mnistANN.longANN()

    elif n == 2:
        print(auxLine)
        print("\nFunção de ativação ('identity', 'logistic', 'tanh', 'relu'):")
        activation = input()

        print("\nResolvedor ('lbfgs', 'sgd', 'adam'):")
        solver = input()

        print("\nHidden Layer(s) (Insira a quantidade, separada por espaços) -> '100 50':")
        layers = input()
        
        print()

        mnistANN.shortAnn(activation=activation, solver=solver, nHiddenLayers=layers)

    else:
        print("Valor inválido")

# GB
elif n == 2:
    mnistGB.shortGB()

else:
    print("Valor inválido")
    exit()

print("Terminando a aplicação! Espero que tenha se divertido =).")