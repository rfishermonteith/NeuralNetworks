# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 15:09:05 2018

@author: richardf
"""
import cPickle, gzip, numpy as np, matplotlib.pyplot as plt, time


# Load the MNIST data
f = gzip.open('../mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

plt.ion()
fig = plt.figure()

#ax = fig.add_subplot(111)

# Display an image
for k in range(0,10):
    pixels = train_set[0][k].reshape(28,28)
    # pixels = np.array(pixels, dtype='uint8').reshape(28,28)
    plt.imshow(pixels,cmap='gray')
    plt.show()
    fig.canvas.draw()
    print("Updated")
    time.sleep(0.25)
    