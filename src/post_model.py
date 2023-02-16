import joblib
import numpy as np
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

# from sklearn.ensemble import RandomForestClassifier  # also try this


class PostModel:
    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self.load()

    def train(self, X: np.ndarray, y: np.ndarray):
        self._model = DecisionTreeClassifier()
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self):
        if self._model is not None:
            joblib.dump(self._model, self._model_path)
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load(self):
        try:
            self._model = joblib.load(self._model_path)
        except:
            self._model = None
        return self

    def load_training_data(self, path="post_training.csv"):
        """
        :param path: path to csv holding training data.
        :return:  X,Y  np.ndarray shape (width,n_examples), np.ndarray shape (n_examples)
        """
        # load csv where one col. is y and rest of cols are X
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        nrows = data.shape[0]
        print(f"  loaded {nrows} of training data")
        ncols = data.shape[1]
        # last col is y
        y = data[:, ncols-1]
        x = data[:, 0:ncols-1]
        return x, y

    def get_model(self):
        return self._model


#
#   train the model
#
if __name__ == "__main__":
    n_arguments = len(sys.argv) - 1
    if n_arguments != 2:
        print("usage: training_data_csv model_save")
        sys.exit(1)
    training_data_path = sys.argv[1]
    model_save_path = sys.argv[2]
    model = PostModel(model_save_path)
    X, y = model.load_training_data(training_data_path)
    print(f"  loaded {training_data_path} with {X.shape[1]} features and {X.shape[0]} examples")

    # do cross-validation to get an idea of how sensitive inference is to shuffling and picking
    model_t = model.get_model()
    for i, score in enumerate(cross_validate(DecisionTreeClassifier(), X, y, cv=5)["test_score"]):
        print(f"Accuracy for the fold no. {i} on the test set: {score}")

    # final training with all data
    model.train(X, y)
    model.save()
    print(f"  saved model in {model_save_path}")
    # measure fit
    print(f"Accuracy on Train: {accuracy_score(y, model.predict(X))}")

    #
    #            visualize the decision tree
    #
    feature_names = ['predicted_label_1_0', 'region_size_1_0', 'confidence_1_0', 'aspect_ratio_1_0', 'normalized_x_1_0',
                     'normalized_y_1_0', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_1_1', 'region_size_1_1', 'confidence_1_1', 'aspect_ratio_1_1', 'normalized_x_1_1',
                     'normalized_y_1_1', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_1_2', 'region_size_1_2', 'confidence_1_2', 'aspect_ratio_1_2', 'normalized_x_1_2',
                     'normalized_y_1_2', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_1_3', 'region_size_1_3', 'confidence_1_3', 'aspect_ratio_1_3', 'normalized_x_1_3',
                     'normalized_y_1_3', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_2_0', 'region_size_2_0', 'confidence_2_0', 'aspect_ratio_2_0', 'normalized_x_2_0',
                     'normalized_y_2_0', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_2_1', 'region_size_2_1', 'confidence_2_1', 'aspect_ratio_2_1', 'normalized_x_2_1',
                     'normalized_y_2_1', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_2_2', 'region_size_2_2', 'confidence_2_2', 'aspect_ratio_2_2', 'normalized_x_2_2',
                     'normalized_y_2_2', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_2_3', 'region_size_2_3', 'confidence_2_3', 'aspect_ratio_2_3', 'normalized_x_2_3',
                     'normalized_y_2_3', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_3_0', 'region_size_3_0', 'confidence_3_0', 'aspect_ratio_3_0', 'normalized_x_3_0',
                     'normalized_y_3_0', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_3_1', 'region_size_3_1', 'confidence_3_1', 'aspect_ratio_3_1', 'normalized_x_3_1',
                     'normalized_y_3_1', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_3_2', 'region_size_3_2', 'confidence_3_2', 'aspect_ratio_3_2', 'normalized_x_3_2',
                     'normalized_y_3_2', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_3_3', 'region_size_3_3', 'confidence_3_3', 'aspect_ratio_3_3', 'normalized_x_3_3',
                     'normalized_y_3_3', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_4_0', 'region_size_4_0', 'confidence_4_0', 'aspect_ratio_4_0', 'normalized_x_4_0',
                     'normalized_y_4_0', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_4_1', 'region_size_4_1', 'confidence_4_1', 'aspect_ratio_4_1', 'normalized_x_4_1',
                     'normalized_y_4_1', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_4_2', 'region_size_4_2', 'confidence_4_2', 'aspect_ratio_4_2', 'normalized_x_4_2',
                     'normalized_y_4_2', 'page_width_fraction', 'page_height_fraction',
                     'predicted_label_4_3', 'region_size_4_3', 'confidence_4_3', 'aspect_ratio_4_3', 'normalized_x_4_3',
                     'normalized_y_4_3', 'page_width_fraction', 'page_height_fraction']
    dotfile = open("dt.dot", 'w')
    class_names = ["other", "start_article", "refs", "toc"]
    tree.export_graphviz(model.get_model(), out_file=dotfile, feature_names=feature_names, class_names=class_names)
    dotfile.close()
    print(f" use http://webgraphviz.com/  to view the dt.dot file")
