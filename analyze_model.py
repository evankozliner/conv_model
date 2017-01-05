import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import helpers

if __name__ == "__main__":
    m = np.loadtxt('tf_log_out.csv', dtype=(int,int), delimiter= ",")
    predicted = m[:,0]
    actual = m[:,1]
    plt.figure
    conf = confusion_matrix(actual, predicted)
    helpers.print_confusion_scores(conf)
    helpers.plot_confusion_matrix(conf, list(set(actual)))
    plt.show()
