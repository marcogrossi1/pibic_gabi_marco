# MNIST Solver

Watch the video tutorial: 

[Portuguese Version](https://drive.google.com/file/d/1it3432Ej-O-cSWLGwf0YwDMscI4iJnLb/view?usp=sharing).
[English Version](https://drive.google.com/file/d/1qZbmIKdIAw6m02JFhiIlg-XbeSS595C6/view?usp=sharing).

This is a simple command-line application to solve the **MNIST handwritten digit classification** problem using two different machine learning approaches:

- **Artificial Neural Network (ANN)**
- **Gradient Boosting (GB)**

Whether you want to test models quickly or do a full parameter search, this app supports both.

---

## Getting Started

### 1. Download and Extract

Download the entire repository and extract it into a local folder.

### 2. Run the Application

Navigate to the `MNIST` folder and run:

```bash
python app.py
```

The application will automatically install all required dependencies (this can take up to 2 minutes):

- `numpy`
- `scikit-learn`
- `pandas`
- `seaborn`
- `matplotlib`

Then, the program will start running.

---

## How to Use

After launching, you will be prompted to choose between two machine learning methods:

1. **Artificial Neural Network (ANN)**

   You can select:

   - **Parameter Search (estimated 30 minutes)**  
     Performs an extended search for the best parameters.

   - **Quick Run (estimated 45 seconds)**  
     You will be asked to input:
     - Activation function (`identity`, `logistic`, `tanh`, `relu`)
     - Solver (`lbfgs`, `sgd`, `adam`)
     - Hidden layer sizes (e.g., `100 50` for two layers with 100 and 50 neurons)

2. **Gradient Boosting (GB)**

   Runs a Gradient Boosting classifier on the MNIST dataset (estimated 30 minutes).

---

## Important Notes

- Please type your inputs carefully.  
- If you enter an invalid option or typo, the program will stop.  
- If this happens, just restart the application by running `app.py` again.

---

## Example Interaction
Olá! Vamos resolver o problema dos dígitos do MNIST utilizando métodos de ML!!!
Qual método você deseja utilizar?

1.Artificial Neural Network - ANN

2.Gradient Boosting - GB

    1

Você prefere testar os parametros?

1.Sim, quero encontrar os melhores parametros dentre os disponíveis (tempo estimado 30 min). 

2.Não, prefiro rodar com padrões loucos e fé (tempo estimado 45 seg)

    2

Função de ativação ('identity', 'logistic', 'tanh', 'relu'):

    relu

Resolvedor ('lbfgs', 'sgd', 'adam'):

    adam

Hidden Layer(s) (Insira a quantidade, separada por espaços) -> '100 50':

    100 50

---

## Enjoy!

The app will train your model and show results including accuracy and prediction examples.

Feel free to contribute or report issues!