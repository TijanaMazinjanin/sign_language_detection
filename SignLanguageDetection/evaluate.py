import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load data
DATA_FILE_PATH = './data.pickle'
dict = pickle.load(open(DATA_FILE_PATH, 'rb'))
signes = np.asarray(dict['signes'])
data = np.asarray(dict['data'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, signes, test_size=0.2, shuffle=True, stratify=signes)

# Load models
model_dict = pickle.load(open('model.p', 'rb'))
model_knn = model_dict['modelKNN']
model_rf = model_dict['modelRF']
model_svc = model_dict['modelSVC']
model_cnn = model_dict['modelCNN']

# Evaluation function
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, matrix

def evaluate_cnn_model( ):
    label_encoder = LabelEncoder()
    signs_encoded = label_encoder.fit_transform(signes)
    num_classes = len(np.unique(signs_encoded))
    num_coordinates = 84
    data_ = data.reshape(-1, num_coordinates, 1)
    x_train_cnn, x_test_cnn, y_train_cnn, y_test_cnn = train_test_split(data_, signs_encoded, test_size=0.2, shuffle=True, stratify=signs_encoded)
    y_test_cnn = to_categorical(y_test_cnn, num_classes)
    y_pred = model_cnn.predict(x_test_cnn)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_cnn, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')


    return accuracy, precision, recall

# Plotting function
def plot_results(results):
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    precisions = [results[model]['precision'] for model in models]
    recalls = [results[model]['recall'] for model in models]

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    #Plot accuracy
    axs[0].bar(models, accuracies, color=['blue', 'cyan', 'green', 'magenta'], alpha=0.7)
    axs[0].set_ylim(0.8, 1.1)
    axs[0].set_title('Accuracy')
    axs[0].set_ylabel('Score')

    # Plot precision
    axs[1].bar(models, precisions, color=['blue', 'cyan', 'green', 'magenta'], alpha=0.7)
    axs[1].set_ylim(0.8, 1.1)
    axs[1].set_title('Precision')
    axs[1].set_ylabel('Score')

    # Plot recall
    axs[2].bar(models, recalls, color=['blue', 'cyan', 'green', 'magenta'], alpha=0.7)
    axs[2].set_ylim(0.8, 1.1)
    axs[2].set_title('Recall')
    axs[2].set_ylabel('Score')


    plt.tight_layout()
    plt.show()

# Evaluate models and collect results
results = {
    'Random Forest': {
        'accuracy': evaluate_model(model_rf, x_test, y_test)[0],
        'precision': evaluate_model(model_rf, x_test, y_test)[1],
        'recall': evaluate_model(model_rf, x_test, y_test)[2],
        'matrix': evaluate_model(model_rf, x_test, y_test)[3],
    },
    'K Nearest Neighbors': {
        'accuracy': evaluate_model(model_knn, x_test, y_test)[0],
        'precision': evaluate_model(model_knn, x_test, y_test)[1],
        'recall': evaluate_model(model_knn, x_test, y_test)[2],
        'matrix': evaluate_model(model_knn, x_test, y_test)[3]
    },
    'Support Vector Classifier': {
        'accuracy': evaluate_model(model_svc, x_test, y_test)[0],
        'precision': evaluate_model(model_svc, x_test, y_test)[1],
        'recall': evaluate_model(model_svc, x_test, y_test)[2],
        'matrix': evaluate_model(model_svc, x_test, y_test)[3]
    },
    'CNN': {
        'accuracy': evaluate_cnn_model()[0],
        'precision': evaluate_cnn_model()[1],
        'recall': evaluate_cnn_model()[2]
    }
}
print(results)
# Plot results
plot_results(results)
