import matplotlib.pyplot as plt

def plot_prediction(y_true, y_pred, title):
    plt.figure(figsize=(14,5))
    plt.plot(y_true, label='Réel')
    plt.plot(y_pred, label='Prédit')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
