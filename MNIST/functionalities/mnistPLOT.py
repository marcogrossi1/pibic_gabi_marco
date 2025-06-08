from sklearn.metrics import confusion_matrix

from numpy import hstack
from random import random

import seaborn as sns
import matplotlib.pyplot as plt

def plotDigits(xTest, model, yTest):
    xTest_np = xTest.values if hasattr(xTest, 'values') else xTest
    yTest_np = yTest.values if hasattr(yTest, 'values') else yTest

    num_samples = 5
    digit_images = []
    labels = []

    ranValue = int(random() * 1000)

    for i in range(ranValue, ranValue + num_samples):
        img = xTest_np[i].reshape(28, 28)
        pred = model.predict(xTest_np[i].reshape(1, -1))[0]
        actual = yTest_np[i]
        digit_images.append(img)
        labels.append(f"P:{pred} | A:{actual}")

    combined_image = hstack(digit_images)

    plt.figure(figsize=(num_samples * 2, 2.5))
    plt.imshow(combined_image, cmap='gray')
    plt.axis('off')
    plt.title('   '.join(labels), fontsize=12)
    plt.show()

def plotConfusion(yTest, preds):
    cm = confusion_matrix(yTest, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()