import numpy as np

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def identity_function(x) : # 입력의 그대로를 출력하는 함수. 출력층의 활성화 함수로 이용
    # 풀고자 하는 문제의 성질에 맞게 정하면 됨
    # 회귀 : 항등함수
    # 2 클래스 분류 = 시그모이드
    # 다중 클래스 분류 = 소프트맥스함수를 사용하는 것이 일반적
    return x

def init_network() :
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x) : # 입력 -> 출력으로 전달 과정, 신호가 순방향으로 전달
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)