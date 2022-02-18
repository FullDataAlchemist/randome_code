import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay


def plotting_conf_2(predictions_max, y_test, class_names, result_name, plot_path):
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions_max, axis=1))
    # cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
    #                       labels=labels, normalize=normalize)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(include_values=True,
              cmap='viridis', ax=None, xticks_rotation='horizontal',
              values_format=None)
    plt.savefig(plot_path + "confusion_matrix2-" + result_name)


def ploting_confusion_matrix(predictions_max, y_test, class_names, result_name, plot_path):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions_max, axis=1))
    print(cm)
    fig, ax = plt.subplots(figsize=(37, 37))
    # plt.rcParams.update({'font.size': 5})
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()


    ax.xaxis.set_ticklabels(class_names, rotation=45)
    ax.yaxis.set_ticklabels(class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(plot_path +"confusion_matrix-" + result_name.replace('.hdf5',''))


def plot_acc_loss(history, result_name: str, plot_path: str):
    plt.style.use('ggplot')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy, Test is: {:0.3f}'.format(max(val_acc)))
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.legend()
    plt.savefig(plot_path + "/plot-" + result_name)


def log_best_scores(history, result_name, plot_path):
    accr = history.history['accuracy']
    tmp = ''
    tmp += result_name + ';'
    tmp += str(np.max(accr)) + ';'
    tmp += str(np.max(history.history['val_accuracy'])) + ';'
    tmp += str(accr[-1]) + '\n'
    with open(plot_path + '/log_best_scores.txt', 'a') as out_result:
        out_result.write(tmp)
