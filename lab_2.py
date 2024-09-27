import numpy as np

W = np.array(([5, -2], [9, 73]), dtype=float)
W2 = np.array(([4], [12]), dtype=float)
W3 = np.array(([10]),dtype=float)
b1 = 2
b2 = -3
b4 = 4
x = np.array(([3, 2]), dtype=float)

output1 = np.dot(x, W) + b1
output1

def sigmoid(val):
    return 1/(1+np.exp(-val))
predictios = sigmoid(x)
predictios

predictios  = predictios * 100
predictios  

output_2 = np.dot(output1, W2) + b2
output_2

output_3 = np.dot(output_2 , W3) + b4
output_3


# Outputs
print("Output of Layer 1:", predictios)
print("Output of Layer 2:", output_2)
print("Output of Layer 3:", output_3)