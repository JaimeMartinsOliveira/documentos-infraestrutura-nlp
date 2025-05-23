from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, labels=None):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision (macro):", precision_score(y_true, y_pred, average='macro', zero_division=0))
    print("Recall (macro):", recall_score(y_true, y_pred, average='macro', zero_division=0))
    print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro', zero_division=0))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0, target_names=labels))


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 2, 2, 2, 1]
    labels = ['Classe 0', 'Classe 1', 'Classe 2']

    evaluate_model(y_true, y_pred, labels)
    plot_confusion_matrix(y_true, y_pred, labels)
