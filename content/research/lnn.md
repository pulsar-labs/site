+++
author = "Tobias Jone"
title = "L-NN for IoT"
date = "2023-10-10"
description = "L-NN - a lightweight non-parametrised supervised-learning algorithm for IoT applications"
tags = [
    "research",
    "IoT",
    "machine learning"
]
+++
# White Paper - Application of a Lightweight Distance-Based Classification Algorithm in IoT Environments
By: Tobias Jone

**Abstract**

This paper explores the application of Light Nearest Neighbours (L-NN), a lightweight distance-based classification algorithm, in the domain of Internet of Things (IoT). L-NN, originally designed for nano-satellite technology, offers an efficient solution for classifying data captured by IoT sensor systems. The paper introduces L-NN and discusses its potential applications in various IoT scenarios.

## Introduction

In this paper, we investigate the design and application of L-NN within the context of Internet of Things (IoT). With the increasing prevalence of IoT devices across diverse industries, there is a growing need for lightweight and resource-efficient classification algorithms suitable for IoT environments.

## Background

IoT devices, ranging from industrial sensors to weather balloons, operate in varied environments with limited computational resources. Traditional classification algorithms may pose challenges in such resource-constrained scenarios. The L-NN algorithm, originally designed for nano-satellite technology, proves to be well-suited for addressing the unique challenges of IoT applications.

## Method

L-NN, a lightweight classification solution, is tailored to the resource constraints present in IoT devices. Similar to its application in nano-satellite technology, L-NN leverages distance-based classification to categorise input data captured by various sensor systems in IoT environments.

The key difference between L-NN and traditional k-NN lies in L-NN's ability to pre-compress the dataset into a new model, containing a single vector per target class. This compression significantly reduces computation time, power consumption, and storage space requirements, making it an ideal choice for IoT devices.

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

    Initialise an empty list `L_l` to store the feature vectors of data points with label `l`.

    Initialise a count variable `count_l` to count the number of data points with label `l`.

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

### Data Compression

In the context of IoT, where devices often operate with limited bandwidth and power resources, data compression is crucial. L-NN's adaptation for IoT efficiently compresses data before classification, reducing the data load transmitted both from and to IoT devices.

### Applications in IoT

L-NN can be applied in various IoT scenarios, including:

- **Industrial Sensors:** Efficient classification of data from industrial sensors for monitoring and control purposes.
- **Environmental Monitoring:** Classification of environmental data, such as weather parameters collected by IoT weather balloons.
- **Real-time Decision-Making:** Lightweight classification for on-board decision-making in IoT devices, optimizing mission objectives and resource usage.

## Advantages over Traditional AI/CNN Models in IoT

### Resource Efficiency

In comparison to AI and CNN approaches, which may demand substantial computational resources, L-NN's lightweight nature makes it more suitable for IoT devices with limited processing power and energy.

### Bandwidth Conservation

IoT devices often operate with constrained bandwidth for data transmission. L-NN's data compression capabilities significantly reduce the volume of data transmitted, conserving precious bandwidth and minimizing power consumption.

### Real-Time Decision-Making

IoT devices frequently require rapid decision-making. L-NN's simplicity and speed make it well-suited for real-time onboard decision-making, allowing IoT devices to adjust their operations autonomously.

### Adaptability to Variable Environments

IoT devices operate in dynamic and often unpredictable environments. L-NN's distance-based approach allows it to adapt easily to changing circumstances without the need for frequent model updates.

## Results and Future Directions

Preliminary experiments in a simulated IoT environment show promise for L-NN in terms of resource-efficient classification. Ongoing research aims to further optimise L-NN for specific IoT applications, explore advanced distance metrics tailored to IoT data, and enhance classification performance in high-dimensional datasets.

## Conclusion

The application of distance-based classification algorithms, such as L-NN, holds potential in revolutionizing the way IoT devices handle data classification and decision-making. As IoT technology continues to advance, innovative techniques like L-NN will play a crucial role in optimizing resource utilization, enhancing operational efficiency, and expanding the possibilities of IoT applications.

