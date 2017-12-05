
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:

# Load pickled data
import pickle
from random import shuffle
# TODO: Fill this in based on where you saved the training and testing data

training_file = "traffic-signs-data/train.p"
validation_file= "traffic-signs-data/valid.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
assert(len(X_train) == len(y_train))

X_valid, y_valid = valid['features'], valid['labels']
assert(len(X_valid) == len(y_valid))


X_test, y_test = test['features'], test['labels']
assert(len(X_test) == len(y_test))


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[2]:

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np
import pandas as pd

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image? It is (32, 32, 3) -> (width, height, depth)
image_shape = np.shape(X_train[0])

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[3]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
label = pd.read_csv('signnames.csv')
plt.figure(figsize=(25, 25))
plt.subplots_adjust( hspace=.4, wspace=.4 )
for i in range(0, n_classes):
    plt.subplot(11, 4, i + 1)
    for j in range(len(y_train)):
        if i == y_train[j]:
            img = X_train[j]
            break

    plt.imshow(img.squeeze()) 
    plt.title(str(label.loc[y_train[j], 'SignName']), fontsize=20)
    plt.axis('off')
plt.show()
plt.clf()
plt.cla()
plt.close()



# * ** plot distribution of different dataset**

# In[4]:

# print distribution of traffic signs in training set
print("---> Message: distribution of training set")
num_bins = 100
plt.figure()
n, bins, patches = plt.hist(y_train, num_bins, facecolor='blue', alpha=0.5)
plt.show()
plt.clf()
plt.cla()
plt.close()
# print distribution of traffic signs in validation set
print("---> Message: distribution of validation set")
num_bins = 100
plt.figure()
n, bins, patches = plt.hist(y_valid, num_bins, facecolor='blue', alpha=0.5)
plt.show()
plt.clf()
plt.cla()
plt.close()
# print distribution of traffic signs in testing set
print("---> Message: distribution of testing set")
num_bins = 100
plt.figure()
n, bins, patches = plt.hist(y_test, num_bins, facecolor='blue', alpha=0.5)
plt.show()
plt.clf()
plt.cla()
plt.close()


# * Show specific number of images in each category

# In[5]:

num_img_per_class = np.zeros(n_classes)

# count number of image in each category
for i in range(0, len(y_train)):
    num_img_per_class[y_train[i]] += 1

# print statistic info
for i in range(0, n_classes):
    print("{:2d}".format(i), "\t count := " + str(num_img_per_class[i]),
          "\t traffic sign name := " + label.loc[i, 'SignName'], 
          )


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# * **convert RGB image into gray image**

# In[6]:

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
import matplotlib.image as mpimg

# Y' = 0.299 R + 0.587 G + 0.114 B 
def rgb2gray(img):
    return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

#print(X_train[0].shape)
# convert training RGB images into grayscale
X_train_gray = []
for i in range(len(X_train)):
    gray = rgb2gray(X_train[i])
    X_train_gray.append(np.stack((gray,)*1, axis=2))
assert(len(X_train) == len(X_train_gray))  
#print(X_train_gray[0].shape)

# convert validation RGB images into grayscale 
X_valid_gray = []
for i in range(len(X_valid)):
    gray = rgb2gray(X_valid[i])
    X_valid_gray.append(np.stack((gray,)*1, axis=2))
assert(len(X_valid) == len(X_valid_gray))

# convert testing RGB images into grayscale
X_test_gray = []
for i in range(len(X_test)):
    gray = rgb2gray(X_test[i])
    X_test_gray.append(np.stack((gray,)*1, axis=2))
assert(len(X_test) == len(X_test_gray))

print("Message: dimension of image change from " + str(np.shape(X_train[0])) + " to " +
      str(np.shape(X_train_gray[0])) + " after rgb2gray transform")
# print one example
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()


# * ** plot original image and grayscale image for comparision; randomly pick 10 images **

# In[7]:

import random

# plot original image "X" and corresponding grayscale image "Y" in the same row. pick K images randomly
def plot_image_cmp(X, Y, k = 10):
    f, axarr = plt.subplots(k, 2, figsize=(32, 32))
    # generate k indices for image to plot
    indices = random.sample(range(0, n_train), k)
    count = 1
    for i in range(len(indices)):
            image = X[indices[i]].squeeze()
            gray = Y[indices[i]].squeeze()
            plt.subplot(k, 2, count)
            plt.imshow(image)
            plt.subplot(k, 2, count + 1)
            plt.imshow(gray, cmap = plt.get_cmap('gray'))
            count+=2
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
        
plot_image_cmp(X_train, X_train_gray, 10)


# * **scale the image data so that the data has mean zero and equal variance.**

# In[8]:

# scale the data
X_train_scale = []
for i in range(len(X_train)):
    X_train_scale.append(( X_train_gray[i] - 128 ) / 128)
assert(len(X_train) == len(X_train_scale))
    
X_valid_scale = []
for i in range(len(X_valid)):
    X_valid_scale.append(( X_valid_gray[i] - 128 ) / 128)
assert(len(X_valid) == len(X_valid_scale))

X_test_scale = []
for i in range(len(X_test)):
    X_test_scale.append(( X_test_gray[i] - 128 ) / 128)
assert(len(X_test) == len(X_test_scale))

# compare images before and after scaling
plot_image_cmp(X_train, X_train_scale, 10)


# Image pixels can take values determined by the dtype of the image, such as 0 to 255 for uint8 images or [0,1] for floating-point images. However, most images either have a narrower range of values (because of poor contrast), or have most pixel values concentrated in a subrange of the accessible values. skimage.exposure provides functions that spread the intensity values over a larger range.
# 
# contrast enhancement rescale_intensity() consists in stretching pixel values to the whole allowed range, using a linear transformation

# In[9]:

from skimage import exposure

# Contrast stretching
X_train_expo = []
for i in range(len(X_train)):
    X_train_expo.append(exposure.rescale_intensity(X_train_scale[i]))
assert(len(X_train) == len(X_train_expo))
    
X_valid_expo = []
for i in range(len(X_valid)):
    X_valid_expo.append(exposure.rescale_intensity(X_valid_scale[i]))
assert(len(X_valid) == len(X_valid_expo))

X_test_expo = []
for i in range(len(X_test)):
    X_test_expo.append(exposure.rescale_intensity(X_test_scale[i]))
assert(len(X_test) == len(X_test_expo))

plot_image_cmp(X_train, X_train_expo, 5)


# In[10]:

id = 25
plt.figure()
plt.imshow(X_train[id].squeeze())
plt.show()

plt.imshow(X_train_gray[id].squeeze(), cmap = plt.get_cmap('gray'))
plt.show()

plt.imshow(X_train_scale[id].squeeze(), cmap = plt.get_cmap('gray'))
plt.show()

plt.imshow(X_train_expo[id].squeeze(), cmap = plt.get_cmap('gray'))
plt.show()

plt.clf()
plt.cla()
plt.close()


# * **Shuffle the data**

# In[11]:

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train_expo, y_train)
X_validation, y_validation = shuffle(X_valid_expo, y_valid)
X_test, y_test = shuffle(X_test_expo, y_test)


# ### Model Architecture

# * Setup tensorflow

# In[12]:

import tensorflow as tf


# * Feature and Labels

# In[13]:

x = tf.placeholder(tf.float32, (None, X_train[0].shape[0], X_train[0].shape[1], 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)


# * Define Convolutional Neural Network

# In[97]:

# (1) initialize weights and bias
mu = 0
sigma = 0.1
filter_size = 5
conv1_depth = 64
conv2_depth = 32
fc1_out = 120
fc2_out = 84
output_size = n_classes

weights = {
    'conv1_W': tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, 1, conv1_depth), mean = mu, stddev = sigma)),
    'conv2_W': tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, conv1_depth, conv2_depth), mean = mu, stddev = sigma)),
    'fc1_W': tf.Variable(tf.truncated_normal(shape=(filter_size*filter_size*conv2_depth, fc1_out), mean = mu, stddev = sigma)),
    'fc2_W': tf.Variable(tf.truncated_normal(shape=(fc1_out, fc2_out), mean = mu, stddev = sigma)),
    'fc3_W': tf.Variable(tf.truncated_normal(shape=(fc2_out, output_size), mean = mu, stddev = sigma))  
}

biases = {
    'conv1_b': tf.Variable(tf.zeros(conv1_depth)),
    'conv2_b': tf.Variable(tf.zeros(conv2_depth)),
    'fc1_b': tf.Variable(tf.zeros(fc1_out)),
    'fc2_b': tf.Variable(tf.zeros(fc2_out)),
    'fc3_b': tf.Variable(tf.zeros(output_size))
}


# In[98]:

from tensorflow.contrib.layers import flatten

# (2)define convolutional layer and fully_connected layer
def convolution_layer(x, weight, bias, relu = True, maxpool = False, valid = True):
    # do convolution with different padding methods
    #print("weight size:") 
    #print(weight.get_shape())
    #print()
    #print("x input size:")
    #print(x.get_shape())
    #print()
    
    if valid:
        tf.Print(weight, [weight])
        res = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')
    else:
        res = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')
    res = tf.add(res, bias)
    
    #print("conv2d result size:")
    #print(res.get_shape())
    #print()
    
    # do relu if true
    if relu:
        res = tf.nn.relu(res)
    # do maxpool if true
    if maxpool:
        if valid:
            res = tf.nn.max_pool(res, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        else:
            res = tf.nn.max_pool(res, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return res

# define fully-connected layer
def fully_connected_layer(x, weight, bias, relu = True, need_flatten = True, dropout = True, keep_prob = 0.75):
    if need_flatten:
        x = flatten(x)
    res = tf.add( tf.matmul(x, weight), bias )
    if relu:
        res = tf.nn.relu(res)
    if dropout:
        res = tf.nn.dropout(res, keep_prob)
    return res


# In[99]:

# (3) Define your sequential architecture here.
def neural_network(x, weights, biases, dropout_prob = 0.75):
    # first convolutional layer (Input = 32x32x1. conv2d_Output = 28x28x6. maxpool_output= 14x14x6.)
    c1 = convolution_layer(x, weights['conv1_W'], biases['conv1_b'], relu = True, maxpool = True, valid = True)
    
    # second convolutional layer (Input = 14x14x6. conv2d_Output = 10x10x16. maxpool_output= 5x5x16.)
    c2 = convolution_layer(c1, weights['conv2_W'], biases['conv2_b'], relu = True, maxpool = True, valid = True)
    
    # first fully-connected layer ( Flatten. Input = 5x5x16. Output = 400.)
    # Input = 400. Output = 120. - need flatten once right here!
    f1 = fully_connected_layer(c2, weights['fc1_W'], biases['fc1_b'], keep_prob = dropout_prob)
    
    
    # second fully-connected layer (Input = 120. Output = 84.)
    f2 = fully_connected_layer(f1, weights['fc2_W'], biases['fc2_b'], need_flatten = False, keep_prob = dropout_prob)
    
    # third fully-connected layer -> output layer (Input = 84. Output = 43.)
    output = fully_connected_layer(f2, weights['fc3_W'], biases['fc3_b'], need_flatten = False, relu = False, dropout = False)
    # return
    return output


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# * Training Pipeline

# In[100]:

rate = 0.001
EPOCHS = 50
BATCH_SIZE = 256

logits = neural_network(x, weights, biases, dropout_prob = 1.0)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# * Model Evaluation

# In[101]:

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# * Train the model

# In[102]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        """
        shuffle your data after each epoch because you will always have the risk to create batches 
        that are not representative of the overall dataset, and therefore, your estimate of the gradient will be off. 
        Shuffling your data after each epoch ensures that you will not be "stuck" with too many bad batches.
        """
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            accuracy, loss = sess.run([training_operation, loss_operation], feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1),"validation accuracy = {:.3f}".format(validation_accuracy), "loss = {:.3f}".format(loss))
        
    saver.save(sess, './lenet')
    print("Model saved")


# * Evaluate the model

# In[103]:

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[104]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import os
from skimage import data
from skimage.transform import resize

images=os.listdir("test_img/")
print(images)

load_images = []
new_images = []
for image in images:
    # skip non-png file
    if "png" not in image:
        continue
    # process png images
    filename = os.path.join("test_img/", image)
    img = mpimg.imread(filename)
    load_images.append(img)
    # resize into (32x32) size
    img = resize(img, (32, 32), mode='reflect')
    new_images.append(img)

plt.figure(figsize=(20,20))
columns = 10
for i, image in enumerate(load_images):
    plt.subplot(len(load_images) / columns + 1, columns, i + 1)
    plt.imshow(image)
    plt.axis('off')
plt.show()
plt.clf()
plt.cla()
plt.close()


# ### Predict the Sign Type for Each Image

# In[105]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

# pre-processing images
img_processed=[]
for i in range(len(new_images)):
    # convert rgb into grayscale
    img_gray = rgb2gray(new_images[i])
    img_gray = np.stack((img_gray,)*1, axis=2)
    # scale image
    img_scale = (( img_gray - 128 ) / 128)
    # enhance contrast
    img_rescale = exposure.rescale_intensity(img_scale)
    # add into list
    img_processed.append(img_rescale)

# set label for those new images
img_label = np.array([33, 12, 14, 15, 22])

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    prediction = sess.run(tf.argmax(logits, axis = 1), feed_dict={x: img_processed, y: img_label})
    print(prediction)
# plot figures with prediction result
plt.figure(figsize=(32, 32))
plt.subplots_adjust( hspace=.4, wspace=.4 )
for i in range(0, len(load_images)):
    plt.subplot(5, 1, i + 1)
    plt.imshow(load_images[i].squeeze()) 
    plt.title("prediction:="+ str(label.loc[prediction[i], 'SignName']) 
              + " \t  true label:=" + str(label.loc[img_label[i], 'SignName']) , fontsize=20)
    plt.axis('off')
plt.show()
plt.clf()
plt.cla()
plt.close()


# ### Analyze Performance

# In[107]:

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

assert len(prediction) == len(img_label)
num_correct = 0
for i, j in zip(prediction, img_label):
    if i == j:
        num_correct += 1
test_accuracy = num_correct / len(prediction)    
print("Test Accuracy = {:.3f}".format(test_accuracy))


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[108]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    result = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k = 5), feed_dict={x: img_processed, y: img_label})
    #print("5 predictions for each image (in each row)")
    #print()
    #print()
    #print("top  5 predictions above) for each image:")
    #for i in range(len(img_label)):
    #    print(i, result[0][i].T)
        
prediction = result[1]
probability = result[0]

for i, (preds, probs, image) in enumerate(zip(prediction, probability, new_images)):
    print('-' * 60) 
    print('\033[1m')
    correct = label.loc[img_label[i], 'SignName']
    print('\t True Label is: = {:>5}:{:<30}'.format(img_label[i], correct))
    print('\033[0m')
    for res, p in zip(preds, probs):
        name = label.loc[res, 'SignName']
        print('{:>5}: {:<30} {:>3.3f}%'.format(res, name, p*100.0))
    print('-' * 60)        
    


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

