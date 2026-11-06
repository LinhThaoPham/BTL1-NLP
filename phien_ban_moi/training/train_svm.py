"""Cấu hình riêng cho người phụ trách SVM; ROS được thực hiện ở module sampling."""
from sklearn.svm import LinearSVC

PARAMETERS = [0.1, 1.0, 10.0]


def make_classifier(c, seed=42):
    return LinearSVC(C=c, random_state=seed, max_iter=20000, dual="auto")
