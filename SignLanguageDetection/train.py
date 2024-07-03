import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

DATA_FILE_PATH = './data.pickle'

dict = pickle.load(open(DATA_FILE_PATH, 'rb'))
signes = np.asarray(dict['signes'])
data = np.asarray(dict['data'])

x_train, x_test, y_train, y_test = train_test_split(data, signes, test_size=0.2, shuffle=True, stratify=signes)


def train_random_forest():
    rain_forest_model = RandomForestClassifier()
    rain_forest_model.fit(x_train, y_train)    
    return rain_forest_model


def train_k_nearest_neighbors():
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(x_train, y_train)
    return knn_model


def train_support_vector():
    svc_model = SVC(kernel='rbf', C=10, gamma='auto')
    svc_model.fit(x_train, y_train)
    return svc_model


def train_cnn():
    label_encoder = LabelEncoder()
    signs_encoded = label_encoder.fit_transform(signes)

    num_classes = len(np.unique(signs_encoded))

    num_coordinates = 84
    print(data)
    data_ = data.reshape(-1, num_coordinates, 1)
    print(data_)
    print(signs_encoded)
    print(data_.shape[0])
    print(signs_encoded.shape[0])

    x_train_cnn, x_test_cnn, y_train_cnn, y_test_cnn = train_test_split(data_, signs_encoded, test_size=0.2, shuffle=True, stratify=signs_encoded)

    y_train_cnn = to_categorical(y_train_cnn, num_classes)
    y_test_cnn = to_categorical(y_test_cnn, num_classes)

    print(y_train_cnn)
    print(y_test_cnn)

    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=(num_coordinates, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train_cnn, y_train_cnn, epochs=10, batch_size=32, validation_data=(x_test_cnn, y_test_cnn))
    
    return model

def save_models():
    rf_model = train_random_forest()
    svc_model = train_support_vector()
    knn_model = train_k_nearest_neighbors()
    cnn_model = train_cnn()

    f = open('model.p', 'wb')
    pickle.dump({'modelKNN': knn_model, 'modelRF': rf_model, 'modelSVC': svc_model, 'modelCNN': cnn_model}, f)
    f.close()


save_models()



