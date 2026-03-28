"""
线性探针 — 从状态向量解码存储的信息

用法:
1. 收集 (state, label) 对
2. 训练线性分类器/回归器
3. 测试: 能否从状态中解码出特定信息 (用户名、偏好等)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def train_classification_probe(
    states: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    训练分类探针: 能否从状态向量预测离散标签?

    参数:
        states: (N, D) 状态向量集合 (已展平)
        labels: (N,) 对应的标签

    返回:
        dict: train_acc, test_acc, model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        states, labels, test_size=test_size, random_state=random_state,
    )

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "model": clf,
    }


def train_regression_probe(
    states: np.ndarray,
    values: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    训练回归探针: 能否从状态向量预测连续值?

    参数:
        states: (N, D) 状态向量集合
        values: (N,) 对应的连续值

    返回:
        dict: train_r2, test_r2, model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        states, values, test_size=test_size, random_state=random_state,
    )

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)

    train_r2 = r2_score(y_train, reg.predict(X_train))
    test_r2 = r2_score(y_test, reg.predict(X_test))

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "model": reg,
    }


def analyze_dimension_importance(
    states: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    分析哪些状态维度对特定信息最重要。

    返回: (D,) 每个维度的重要性分数 (基于线性模型权重)
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(states, labels)

    # 权重绝对值的均值 (多分类时取所有类的均值)
    if clf.coef_.ndim == 1:
        importance = np.abs(clf.coef_)
    else:
        importance = np.abs(clf.coef_).mean(axis=0)

    return importance
