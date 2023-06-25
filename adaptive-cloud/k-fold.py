from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from data_reader.data_reader import get_data
from models.get_model import get_model


model = get_model(model_name)
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)

x, y =  datasets.load (return_x_y=True)

k_folds = Kfold(n_splits=5)

scores = cross_val_score(clf, x, y, cv =k_folds)


print("Cross Validation Scores: ", scores)
print("Average CV Score: " scores.mean())
print("Number of CV Scores used in Average: ", len(scores))