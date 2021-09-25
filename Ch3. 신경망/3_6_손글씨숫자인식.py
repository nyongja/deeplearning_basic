from re import A
import sys
from matplotlib.pyplot import step
sys.path.append('/Users/nyongja/Library/Mobile Documents/com~apple~CloudDocs/Class/대학원/21-2/deeplearning_basic')
import numpy as np
import pickle
import dataset.mnist as mn
from common.functions import sigmoid, softmax, relu, step_function
import matplotlib.pyplot as plt

def get_data() :
    # (학습 이미지, 학습 레이블), (시험 이미지, 시험 레이블) 형식으로 MNIST 데이터 가져오기
    '''
    normalize : (True) 입력 이미지의 픽셀 값을 0.0 ~ 1.0 사이의 값으로 정규화 
                (False) 0 ~ 255 사이의 값 유지
    flatten : 입력 이미지를 1차원 배열로 저장
            (True) 784개의 원소로 이루어진 1차원 배열로 저장
            (False) 입력 이미지는 1x28x28 3차원 배열로 저장
    one_hot_label : 레이블(정답)을 one-hot encoding 형태로 저장할지 결정.
            (True) one-hot-encoding
            (False) 7, 2, 이런 숫자 형태의 레이블 저장
    '''
    (x_train, t_train), (x_test, t_test) = mn.load_mnist(flatten = True, normalize = False)
    return x_test, t_test

def init_network() :
    with open("./dataset/sample_weight.pkl", 'rb') as f :
        network = pickle.load(f)
    return network


def predict(network, x, f1, f2) :
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    if f1 == "step" :
        z1 = step_function(a1)
    if f1 == "sig" :
        z1 = sigmoid(a1)
    else :
        z1 = relu(a1)

    a2 = np.dot(z1, W2) + b2
    if f2 == "step" :
        z2 = step_function(a2)
    elif f2 == "sig" : 
        z2 = sigmoid(a2)
    else :
        z2 = relu(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network() # 네트워크 생성

batch_size = 100
accuracy_step_step_cnt = 0
accuracy_sig_sig_cnt = 0
accuracy_relu_relu_cnt = 0
accuracy_sig_relu_cnt = 0
accuracy_relu_sig_cnt = 0


for i in range(0, len(x), batch_size) :
    x_batch = x[i:i+batch_size]
    y_batch1 = predict(network, x_batch, 'step', 'step')
    y_batch2 = predict(network, x_batch, 'sig', 'sig')
    y_batch3 = predict(network, x_batch, 'relu', 'relu')
    y_batch4 = predict(network, x_batch, 'sig', 'relu')
    y_batch5 = predict(network, x_batch, 'relu', 'sig')
    
    p1 = np.argmax(y_batch1, axis = 1) # 확률이 가장 높은 인덱스를 얻음
    p2 = np.argmax(y_batch2, axis = 1)
    p3 = np.argmax(y_batch3, axis = 1)
    p4 = np.argmax(y_batch4, axis = 1)
    p5 = np.argmax(y_batch5, axis = 1)

    accuracy_step_step_cnt += np.sum(p1 == t[i:i+batch_size])
    accuracy_sig_sig_cnt += np.sum(p2 == t[i:i+batch_size])
    accuracy_relu_relu_cnt += np.sum(p3 == t[i:i+batch_size])
    accuracy_sig_relu_cnt += np.sum(p4 == t[i:i+batch_size])
    accuracy_relu_sig_cnt += np.sum(p5 == t[i:i+batch_size])

print("Accuracy only step : " + str(float(accuracy_step_step_cnt) / len(x))) 
print("Accuracy only sigmoiod : " + str(float(accuracy_sig_sig_cnt) / len(x))) 
print("Accuracy only relu : " + str(float(accuracy_relu_relu_cnt) / len(x))) 
print("Accuracy sig-relu : " + str(float(accuracy_sig_relu_cnt) / len(x))) 
print("Accuracy relu-sig : " + str(float(accuracy_relu_sig_cnt) / len(x))) 

n = np.arange(5)
values = [float(accuracy_step_step_cnt)/len(x), float(accuracy_sig_sig_cnt)/len(x), float(accuracy_relu_relu_cnt)/len(x), 
            float(accuracy_sig_relu_cnt) / len(x), float(accuracy_relu_sig_cnt) / len(x)]
plt.bar(n, values)
plt.xticks(n, ['step-step', 'sig-sig', 'relu-relu', 'sig-relu', 'relu-sig'])

plt.show()