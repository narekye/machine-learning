import matplotlib.image as plotImg
import matplotlib.pyplot as plt

def showImage(path):
    img = plotImg.inread("../udacity/data/" + path)
    plt.show()