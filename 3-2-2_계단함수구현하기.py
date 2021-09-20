import numpy as np

def step_function(x) :
    if x > 0 :
        return 1
    else :
        return 0

y1 = step_function(1.0)
y2 = step_function(0)
y3 = step_function(-2.0)

print(y1)
print(y2)
print(y3) 