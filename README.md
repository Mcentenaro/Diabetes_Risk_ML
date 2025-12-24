# Diabetes prediction machine learning project
This repository contains my project for the Applied Machine Learning course of the University of Bologna, Bioinformatics MSC.  

## Project summary    
### Data:
data was retrieved from Kaggle at the following link: [https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and it contains 253,680 survey responses from the 2015 Behavioral Risk Factor Surveillance System (BRFSS), a health-related telephone survey that is collected annually by the Centers for Disease Control and Prevention of the United States.  It is a multiclass dataset: class 0 = healthy, class 1 = prediabetes, class 2 = diabetes. More information about data itself can be found in the previous link.  

### Goal:  
Originally the goal was to predict healty, prediabetes or diabetes using a single classifier, but data itself did not allow it since classes highly overlap. I therefore tried an hierarchical strategy employing a first classifier to distinguish between healthy (class 0) and disease (class 1 and class 2) transforming the problem from multiclass to binary, and then a second classifier to distinguish between prediabetes and diabetes. This strategy also failed since class 1 and class 2 cannot, at least with the classifier that I choose, be distinguished.  
Finally, I settled on a binary classifier to distinguish healthy and disease, maximising recall over a more balanced performance, to deliver a classifier useful for screening purposes.  

### Methods:  
The project was fully developed using Python and third-party libraries.    
[Pandas](https://pandas.pydata.org/) and [Numpy](https://numpy.org/) libraries were used to facilitate the storing and visualization of data as well as for data analisys and dataset creation.  
[Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) libraries were used for data visalization.   
[Scikit-learn](https://scikit-learn.org/stable/) and [Imbalance-learn](https://imbalanced-learn.org/stable/) were used to perform cross-validation, model retrival and performance evaluation.  
Other third-party libraries were used for minor tasks.  

### Results:  


