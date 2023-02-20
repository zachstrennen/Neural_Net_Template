"""
FizzBuzz:

For each number in 1 to 100:
-if number is divisible by 3, print "fizz"
-if the number is divisible by 5, print "buzz"
-if the number is divisble by 15, print "fizzbuzz"
-otherwise, print the number
"""

from typing import List

import numpy as np

from Neural_Net_Template.train import train
from Neural_Net_Template.nn import NeuralNet
from Neural_Net_Template.layers import Linear, Tanh
from Neural_Net_Template.optim import SGD


def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_ecode(x: int) -> List[int]:
    """
    10 digit binary encoding of x
    """
    return [x >> i & 1 for i in range(10)]


inputs = np.array([
    binary_ecode(x)
    for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh,
    Linear(input_size=50, output_size=4)
])

train(net,
      inputs,
      targets,
      num_epochs=5000,
      optimizer=SGD(lr=0.001))

for x in range(1, 101):
    predicted = net.forward(binary_ecode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])
