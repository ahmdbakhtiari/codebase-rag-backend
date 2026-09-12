import numpy as np
from services import rag_service

def softmaxConverter(x):
    exp = np.exp(x)
    softmax = exp / np.sum(exp)
    return softmax


def cross_entropy(true, probabilities):
    return -np.sum(true * np.log(probabilities))


