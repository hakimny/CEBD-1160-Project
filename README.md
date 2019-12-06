# CEBD-1160-Project

# Diabetes Dataset

| Name | Date |
|:-------|:---------------|
|Hakim TAMAGLT|December 7, 2019|

-----

### Resources
Your repository should include the following:

- Python script for your analysis: `network_analysis.py`
- Results figure/saved file:  `figures/`
- Dockerfile for your experiment: `Dockerfile`
- runtime-instructions in a file named RUNME.md

-----

## Initial Research Question

Find correlation between age/BMI to determine the illness progress one year after baseline.

## More Refined Research Question

More in depth study of the correlation values between features and the target, gave more insights about how we should refine the reserach question, Based on 

### Abstract

Derived from Diffusion-Weighted Magnetic Resonance Imaging (DWI, d-MRI), we have derived "maps" of structural connectivity between brain regions.
Using these data, we may be able to understand relationships between brain regions and their relative connectivity, which can then be used for targetted interventions in neurodegenerative diseases.
Here, we tried to predict the connectivity between two unique brain regions based on all other known brain connectivity maps.
Based on the preliminary performance of this regressor, we found that the current model didn't provide consistent performance, but shows promise for success with more sophisticated methods.


### Introduction

The graphs used are structural "connectomes" from the publicly available BNU1 dataset([https://neurodata.io/mri-cloud/](https://neurodata.io/mri-cloud/)), processed by Greg Kiar using the ndmg software library [https://github.com/neurodata/ndmg](https://github.com/neurodata/ndmg).
The graphs used here are only a subset of those in the dataset, and in particular only include several of the edges pertaining to the hippocampus and entorhinal cortex for both the left and right hemispheres. 

### Methods

The method used for modelling this data was the Ridge Regressor built into scikit-learn.
Pseudocode (and in particular, the objective function being minimized) can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
Simply put, this objective function minimizes the linear least squares function between predicted and expected value, while being regularized by the L2 norm of the estimated weight matrix.
This method was chosen because of its simplicity.

The data itself was organized into a matrix, and all connections between brain regions with available data were sorted, and then transformed into a table. We attempted to predict the connection found at location 12 in the figure below, from all other connections.

![matrix](./figures/average_graph.png)

### Results

The performance of the regressor was an R^2 value of 0.661. The figure below shows the performance on the testing set.

![performange figure](./figures/performance.png)

We can see that in general, our regressor seems to underestimate our edgeweights. In cases where the connections are small, the regressor performs quite well, though in cases where the strength is higher we notice that the
performance tends to degrade.

### Discussion

The method used here does not solve the problem of identifying the strength of connection between two brain regions from looking at the surrounding regions. This method shows that a relationship may be learnable between these features, but performance suffers when the connection strength is towards the extreme range of observed values. To improve this, I would potentially perform dimensionality reduction, such as PCA, to try and compress the data into a more easily learnable range.

### References
The links referenced were included in my discussion, above.

https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

https://machinelearningmastery.com/linear-regression-for-machine-learning/

https://scikit-learn.org/stable/modules/model_evaluation.html

https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

https://towardsdatascience.com/machine-learning-workflow-on-diabetes-data-part-01-573864fcc6b8

https://realpython.com/pandas-groupby/

https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b

-------
