import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(folder_path, category):
    data = []
    labels = []

    for folder in range(1, 11):
        category_path = os.path.join(folder_path, category, f"part{folder}")

        for filename in os.listdir(category_path):
            with open(os.path.join(category_path, filename), "r", encoding="latin-1") as file:
                content = file.read()
                data.append(content)
                labels.append(1 if filename.startswith("spm") else 0)

    return np.array(data), np.array(labels)

def process_data(data):
    non_empty_docs = [doc for doc in data if doc.strip() != ""]

    if not non_empty_docs:
        raise ValueError("All documents are empty or contain only stop words.")

    vectorizer = CountVectorizer(min_df=2)
    X = vectorizer.fit_transform(non_empty_docs)
    return X

def leave_one_out_cross_validation(X, y, model):
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
    return scores

def plot_cv_results(scores):
    plt.plot(scores)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Leave-One-Out Cross-Validation Results')
    plt.show()

def train_and_test(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

folder_path = "C:/Users/Asus/Documents/UAIC/Anul III/ML/dataset/lingspam_public/lingspam_public"

for category in ["lemm", "stop", "lemm_stop"]:
    data, labels = load_data(folder_path, category)
    non_empty_indices = [idx for idx, doc in enumerate(data) if doc.strip() != ""]
    data = data[non_empty_indices]
    labels = labels[non_empty_indices]
    X = process_data(data)

    X_train = X[:len(labels) - len(labels) // 10]
    y_train = labels[:len(labels) - len(labels) // 10]
    X_test = X[len(labels) - len(labels) // 10:]
    y_test = labels[len(labels) - len(labels) // 10:]

    model = MultinomialNB()  
    accuracy_on_test_set = train_and_test(X_train, y_train, X_test, y_test, model)
    print(f'Accuracy on the test set for category {category}: {accuracy_on_test_set}')

    loo_scores = leave_one_out_cross_validation(X, labels, model)
    plot_cv_results(loo_scores)
