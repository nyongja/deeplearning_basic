import numpy as np
import matplotlib.pylab as plt

def step_function(x) :
    return np.array(x > 0, dtype = np.int32)

# X로 [-5.0, -4.0, ..., 4.9]를 생성
X = np.arange(-5.0, 5.0, 0.1)
y = step_function(X)
print(y)
plt.plot(X,y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()