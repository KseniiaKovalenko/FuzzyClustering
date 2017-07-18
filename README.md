# FuzzyClustering

## General description

Implementation of the clustering algorithm based on the fuzzy equivalence relation and comparing the results of clustering with the algorithms K-Means ++, BIRCH, Spectral Clustering on test datasets and real data. In more detail implemented algorithm based on the fuzzy equivalence relation described in the book [1].

## Requirements

* Python 3
* NumPy
* MatPlotLib
* Sklearn

## Input

1. Test datasets from sklearn.datasets [2].
2. Data on food consumption per capita in 1994 on file data.txt

## Output

Labels of each point and scatter diagram of clustering results.

## References

1. Барсегян А.А., Куприянов М.С., Степаненко В.В., Холод И.И. Методы и модели анализа данных - OLAP и Data Mining (2004), с.190-191.
2. <http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets>
