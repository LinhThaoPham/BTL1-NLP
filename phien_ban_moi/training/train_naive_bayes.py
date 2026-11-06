"""Cấu hình riêng cho người phụ trách Naive Bayes."""
from sklearn.naive_bayes import MultinomialNB

PARAMETERS = [0.1, 0.5, 1.0]


def make_classifier(alpha, seed=42):
    return MultinomialNB(alpha=alpha)
