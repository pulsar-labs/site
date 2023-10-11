+++
author = "Tobias Jone"
title = "L-NN"
date = "2023-10-10"
description = "L-NN - a lightweight non-parametrised supervised-learning algorithm"
tags = [
    "research",
    "IoT",
    "machine learning"
]
+++
# White Paper - Application of a Lightweight Distance-Based Classification-Algorithm in Nano-Satellite Technology
By: Tobias Jone

## Introduction

In this paper, we explore the design and application of a distance-based classification algorithm, Light Nearest Neighbour (L-NN), within the domain of nano-satellite technology. Nano-satellites, with their constrained resources and remote operational environments, present unique challenges that can benefit from innovative classification techniques. 


## Background

Nano-satellites are miniature satellites weighing less than 10kg, and are developed for a wide range of applications, from Earth observation to telecommunications. These small-scale spacecraft often operate in environments where both computational and power resources are limited, making the use of computationally intensive classification algorithms unsuitable for on flight computers, due to their computational and power-consumption demands. Classification is a fundamental task in machine learning, which involves assigning input data to one of several categories. One popular classification algorithm is the k-Nearest Neighbour (k-NN) algorithm, which classifies an input vector by finding the k closest training vectors and assigning the input vector to the category that is most common among the k neighbours. However, the k-NN algorithm can be computationally expensive, especially when dealing with high-dimensional data or datasets with a large number of vectors.

## Method

Our algorithm, L-NN, offers a lightweight classification solution specifically tailored to the constraints present in nano-satellites, and their ability to perform classification tasks within the constrained resources of the deployment environment. L-NN leverages the concept of distance-based classification to categorise input data captured by various sensor systems on-board nano-satellites, with greater CPU efficiency than comparable classification algorithms.

Loosely based around the k-NN algorithm, a non-parametrised supervised-learning algorithm, L-NN  computes the distance between an input vector and a set of training vectors, but in contrast to k-NN, compresses many vectors of a given class into  a single *mean* vector, which requires a fraction of the required compute-time and power resources to yield a classification result (given a typical training dataset, where there are multiple vectors of a given target class). In addition to this, the amount of storage space required to house a model is also far smaller than a comparable k-NN model, which not ony allows for faster result computation, but for updated, or new models, to be transmitted to the flight computer in less time, and with less data bandwidth consumption.

**Implementation of k-NN**


- **Dataset `D`** We have a dataset consisting of data points, each represented as a vector in d-dimensional space. The dataset is denoted as `D`, where `D = {(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)}`, where `x_i` is the feature vector of the i-th data point, and `y_i` is its corresponding label.

- **`k`** k is a user-defined positive integer representing the number of "nearest neighbors" to consider when making predictions.

- **Distance Metric** There is a distance metric function `d(x, y)` that calculates the distance between two feature vectors x and y. Common distance metrics include Euclidean distance, Manhattan distance, etc.


Mathematically, the k-NN algorithm can be expressed as follows for a classification problem:

Given a new data point, `x` for which we wish to make a prediction, the distance is calculated between `x` and all data points in the dataset `D` using the chosen distance metric `d(x, y)`, e.g:

```
Distances = {d(x, x_1), d(x, x_2), ..., d(x, x_N)}
```

The distances are then sorted in ascending order, with the top `k` distances selected, which will be denoted as `Top_k`.

The corresponding labels of the `k` nearest neighbors associated with the `k` smallest distances are retrieved, which will be denoted as `Top_k_Labels`.

Finally, a majority vote among the `k` nearest neighbors' labels is performed, to determine the predicted label for `x`. The predicted label, denoted as `y_pred`, is the label that occurs most frequently among the `Top_k_Labels`, e.g:

```
y_pred = argmax{Count(y_i) for all y_i in Top_k_Labels}
```

Where

`x` is the new data point for which you want to make a prediction.

`x_i` are the feature vectors of the data points in the dataset (`D`).

`y_i` are the labels corresponding to the data points.

`k` is the user-defined parameter specifying the number of neighbors to consider.

`d(x, y)` is the distance metric used to calculate the distance between x and y.

`Distances` is a list of distances between `x` and all data points in the dataset (`D`).
`Top_k` is the list of the `k` smallest distances from `Distances`.

`Top_k_Labels` is the list of labels corresponding to the k nearest neighbors.

`argmax` returns the label `y_pred` that occurs most frequently among the `Top_k_Labels`.

**Implementation of L-NN**

The key difference between k-NN and L-NN is that the L-NN algorithm includes an additional step to pre-compress the dataset into a new model, containing a single vector per target class. 

For example, a dataset `D`, comprised of vectors corresponding to any of 50 target classes, with 100 vectors for each target class, yielding a total of 5000 vectors. With the k-NN algorithm, it would be necessary to iterate over all 5000 vectors, calculating the distance between an unseen vector, `x`, and all vectors in `D`. Using  Using L-NN on the same dataset, `D`, to produce the compressed model, `M`, only 50 distance measurements are performed, regardless of the number of vectors of a given target class.

- **Dataset `D`** We have a dataset consisting of data points, each represented as a vector in d-dimensional space. The dataset is denoted as `D`, where `D = {(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)}`, where `x_i` is the feature vector of the i-th data point, and `y_i` is its corresponding label.

- **Compressed Model (`M`)** L-NN creates a compressed model to represent the dataset. This model, denoted as `M`, stores the mean (average) feature vectors for each unique label in the dataset, and is constructed as follows:

```
For each unique label `l` in the dataset, `D`:

    Initialize an empty list `L_l` to store the feature vectors of data points with label `l`.

    Initialize a count variable `count_l` to count the number of data points with label `l`.

    For each data point `(x_i, y_i)` in the dataset:
    If y_i equals l:
        Add `x_i` to `L_l`.
        Increment `count_l` by `1`.

    Calculate the mean feature vector M_l for label l as the average of feature vectors in L_l.

    Add M_l to the compressed model M with label l.
```

- **Classification**: To classify a new data point `x` into one of the labels, the Euclidean distance is calculated between `x` and each mean feature vector in the compressed model `M`. The label associated with the mean feature vector having the smallest distance to `x` is the predicted label for x.

Mathematically, L-NN can be represented as follows:

**Construction of Compressed Model**

The compressed model M is a dictionary where each key l corresponds to a unique label in the dataset, and the value associated with l is the mean feature vector for that label:

`M = {l_1: M_l1, l_2: M_l2, ..., l_k: M_lk}`

Where:

`l_i` is a unique label in the dataset `D`.
`M_li` is the mean feature vector for label `l_i`.

`M_l` is calculated as:

```
M_l = (1 / count_l) * Σ(x_i) for all x_i in L_l
```

**Classification of a New Data Point `x`**:

Given a new data point `x`, the Euclidean distance is calculated between `x` and each mean feature vector in the compressed model `M`. The predicted label for `x`, denoted as `y_pred`, is the label associated with the mean feature vector having the smallest distance to `x`:

```
y_pred = argmin{d(x, M_l)}, for all l_i in M
```

Where:

`d(x, M_l)` is the Euclidean distance between `x` and the mean feature vector `M_l` for label `l`.
`argmin` returns the label `l_i` for which `d(x, M_li)` is minimised.
 
Below is a Python-like, human-readable implementation of the algorithm. It includes functions to load and compress the dataset, as well as a function to classify a given vector. The magnitude function calculates the Euclidean distance between two vectors. The pseudocode also includes some error checking to ensure that the input data is valid.

```
function load_dataset(path):
    rows = read_csv_file(path)
    dataset, targets = split_rows_into_dataset_and_targets(rows)
    return compress_dataset(dataset, targets)

function compress_dataset(dataset, targets):
    model = {}
    counts = {}

    for i in range(len(targets)):
        t = targets[i]
        d = dataset[i]

        if t not in model:
            model[t] = d
            counts[t] = 1
        else:
            for j in range(len(d)):
                model[t][j] += d[j]
            counts[t] += 1

    _model = {}

    for label in model.keys():
        data = model[label]
        for j in range(len(data)):
            data[j] = data[j] / counts[label]
        _model[label] = data

    targets = []
    dataset = []
    for t, d in _model.items():
        targets.append([t])
        dataset.append(d)

    return dataset, targets


function classify(vector, dataset, outputs):
    if not (vector and dataset and outputs):
        raise ValueError("input data must not be empty")

    if not (len(vector) == len(dataset[0]) and 
            len(dataset) == len(outputs)):
        raise TypeError("data shape must be consistent")

    indexes = []
    for i in range(len(dataset)):
        indexes.append(i)

    distances = []
    for d in dataset:
        distances.append(magnitude(vector, d))

    i = distances.index(min(distances))

    return outputs[i]

function magnitude(a, b):
    distance = 0.0

    for i in range(len(a)):
        distance += (a[i] - b[i]) ** 2

    distance = sqrt(distance)

    return distance
```

## Implementation in Nano-Satellite Technology

### Data Compression

In the context of nano-satellites, where bandwidth and power resources are limited, data compression is critical, and have therefore adapted the L-NN algorithm to compress data efficiently before classification. This adaptation reduces the data load transmitted both from and to nano-satellites, saving valuable resources and time required to perform uplink and downlink of data.

### Remote Sensing and Earth Observation

Nano-satellites play a crucial role in remote sensing and Earth observation, and by applying L-NN, these small spacecraft could potentially classify and transmit valuable data very efficiently. For instance, L-NN can help classify environmental data, such as land cover types or ocean parameters, collected by onboard sensors.

### Onboard Decision-Making

Nano-satellites often operate autonomously in remote locations. L-NN's lightweight classification capabilities can be utilised for onboard decision-making. For example, it can help nano-satellites make real-time decisions about data collection and transmission, optimising mission objectives, or better using available resources.

## The Limitations of AI/CNN in Nano-Satellite Technology

### Resource Intensiveness

AI and CNN approaches, while powerful, often require substantial computational resources. Nano-satellites, operating with limited processing power and energy, may struggle to execute complex AI models efficiently. L-NN's lightweight nature makes it a more viable option, ensuring that classification tasks do not overwhelm the satellite's onboard systems.

### Data Efficiency

Nano-satellites often operate with constrained bandwidth for data transmission. AI/CNN models tend to generate large amounts of data due to their complex architectures. L-NN, with its data compression capabilities, can significantly reduce the volume of data that needs to be transmitted, conserving precious bandwidth and minimising power consumption. Additionally,  because the L-NN target class vectors are compressed into a single vector, the amount of data-transfer required to push new or updated models to a remote system has the potential to be drastically reduced.

### Real-Time Decision-Making

Nano-satellites frequently encounter situations that demand rapid decision-making, such as adapting to changing mission conditions or responding to unexpected events. L-NN's simplicity and speed make it well-suited for real-time onboard decision-making, allowing nano-satellites to autonomously adjust their operations without relying on ground control.

### Adaptability to Variable Environments

Nano-satellites operate in dynamic and often unpredictable environments. AI/CNN models may require extensive retraining when faced with new conditions or data sources. In contrast, L-NN's distance-based approach allows it to adapt more easily to changing circumstances without the need for frequent model updates.

## Harnessing L-NN's Potential

The limitations of AI/CNN models in the context of nano-satellite technology highlight the potential advantages of leveraging L-NN for classification and decision-making tasks



## Results and Future Directions

To evaluate the performance of the algorithm, we applied it to the Iris dataset, comprising three categories: setosa, versicolor, and virginica. We randomly split the dataset into a training set of 120 samples and a test set of 30 samples. We trained the algorithm on the training set and evaluated its performance on the test set.
The algorithm achieved an accuracy of 96.7% on the test set, correctly classifying 29 out of 30 samples. This performance is comparable to that of the k-NN algorithm on the same dataset.


To evaluate the applicability of L-NN in nano-satellite technology, we conducted initial experiments in a simulated satellite environment. While our research is ongoing, preliminary results show promise in terms of resource-efficient classification.

In the future, we aim to further optimise L-NN for specific nano-satellite missions, explore advanced distance metrics tailored to space data, and investigate feature selection techniques to enhance classification performance in high-dimensional datasets.

## Conclusion

The application of distance-based classification algorithms, such as L-NN, holds potential in revolutionising the way nano-satellites handle data classification and decision-making. As nano-satellite technology continues to advance, innovative techniques like L-NN will play a crucial role in optimising resource utilisation, enhancing mission success, and expanding the horizons of space exploration.
