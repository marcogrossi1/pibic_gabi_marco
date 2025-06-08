import language

print("1. Português")
print("2. English")

n = int(input())

if n == 1:
    language.language = "pt"
    import messages.messagesPT as msg

else:
    language.language = "en"
    import messages.messagesEN as msg

print(msg.INSTALLING_DEPENDENCIES)

import importlib.metadata, subprocess, sys

required  = {'numpy','scikit-learn', 'pandas', 'seaborn', 'matplotlib'}
installed = {pkg.metadata['Name'] for pkg in importlib.metadata.distributions()}
missing   = required - installed

if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

print(msg.INITIALIZING)

import methods.mnistANN as mnistANN
import methods.mnistGB as mnistGB

print(msg.DONE)
print(msg.LINE)
print(msg.GREETING)
print(msg.LINE)
print(msg.CHOOSE_METHOD)
print("1. Artificial Neural Network - ANN")
print("2. Gradient Boosting - GB")

n = int(input())

# ANN
if n == 1:
    print(msg.LINE)
    print(msg.CHOOSE_PARAMETERS)
    print(f"1. {msg.YES_PARAM_SEARCH}")
    print(f"2. {msg.NO_PARAM_SEARCH}")

    n = int(input())

    if n == 1:
        mnistANN.longANN()

    elif n == 2:
        print(msg.LINE)
        print(f"\n{msg.CHOOSE_ACTIVATION_FUNCTION}")
        activation = input()

        print(f"\n{msg.CHOOSE_SOLVER}")
        solver = input()

        print(f"\n{msg.CHOOSE_LAYERS}")
        layers = input()
        
        print()

        mnistANN.shortAnn(activation=activation, solver=solver, nHiddenLayers=layers)

    else:
        print(msg.INVALID_VALUE)

# GB
elif n == 2:
    mnistGB.shortGB()

else:
    print(msg.INVALID_VALUE)
    exit()

print(msg.ENDING)