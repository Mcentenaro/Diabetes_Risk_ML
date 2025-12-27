# Diabetes prediction machine learning project
This repository contains my project for the Applied Machine Learning course of the University of Bologna, Bioinformatics MSC.  

## Project summary    
### Data:
data was retrieved from Kaggle at the following link: [https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and it contains 253,680 survey responses from the 2015 Behavioral Risk Factor Surveillance System (BRFSS), a health-related telephone survey that is collected annually by the Centers for Disease Control and Prevention of the United States.  It is a multiclass dataset: class 0 = healthy, class 1 = prediabetes, class 2 = diabetes. More information about data itself can be found in the previous link.  

### Goal:  
Originally the goal was to predict healty, prediabetes or diabetes using a single classifier, but data itself did not allow it since classes highly overlap. I therefore tried an hierarchical strategy employing a first classifier to distinguish between healthy (class 0) and disease (class 1 and class 2) transforming the problem from multiclass to binary, and then a second classifier to distinguish between prediabetes and diabetes. This strategy also failed since class 1 and class 2 cannot, at least with the classifiers that I choose, be distinguished.  
Finally, I settled on a binary classifier to distinguish healthy and disease, maximising recall over a more balanced performance, to deliver a classifier useful for screening purposes.  

### Methods:  
The project was fully developed using Python and third-party libraries.    
[Pandas](https://pandas.pydata.org/) and [Numpy](https://numpy.org/) libraries were used to facilitate the storing and visualization of data as well as for data analisys and dataset creation.  
[Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) libraries were used for data visalization.   
[Scikit-learn](https://scikit-learn.org/stable/) and [Imbalance-learn](https://imbalanced-learn.org/stable/) were used to perform cross-validation, model retrival and performance evaluation.  
Other third-party libraries were used for minor tasks.
##### 1. Data splits:  
Exploratory data analysis revealed high class imbalance:  
| Status | Class label | Abundance |
| --- | --- | --- |
| `Healthy` | 0 | 84.2% |
| `Pre-diabetes` | 1 | 1.8% |
| `Diabetes` | 2 | 13.9% |

Datasets have been divided into a training set, comprising 80% of the dataset, which corresponds to `202945` examples, and a testing set, comprosing 20% of the dataset which corresponds to `50735` examples. The testing set (also known as benchmark set) was only used to assess final model performance.  
Training and benchmark set were created respecting the initial dataset class abundance.  
##### 2. Feature selection:  
Recursive Feature Elimination strategy was employed for feature selection. As classifier, [Balanced Random Forest](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) was used since standard Random Forest failed due to class imbalance.  
##### 3. Model selection:  
The initial cohort of models comprised 7 distinct classifiers:     
| Classifier name | Type | Documentation |
| --- | --- | --- |
| Logistic Regression | regularized logistic regression | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) |
| Balanced Random Forest |  random forest optimized for imbalanced datasets | [imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) |
| Histogram Gradient Boosting | histogram-based gradient boosting classification tree | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) |
| Easy Ensemble | ensemble of AdaBoost learners | [imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html) |
| RUS Boost | AdaBoost with integrated random under-sampling | [imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.RUSBoostClassifier.html) |
| Gaussian Bayes | naive bayes | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) |
| Complement Bayes | naive bayes | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html) |

Recall, escpecially of the pre-diabetes class, and Matthews correlation coefficient (MCC) were used as primary metrics to determine model performance.  
Model selection was performed with a 10-fold cross validation approach on the training set.  
Threshold tuning was performed to maximise recall.  
##### 4. Final training:  
The final model was trained on the entire training set and it's performance evaluated on the benchmark set.  

### Results:
##### 1. Feature selection:
Feature selection did not lead to a reduction of feature to be used on subsequent steps.  
Performance of the balanced random forest (brf) peaked when all features were used.  
Subsequent steps were also performed using a reduced set of 15 features, since brf performance almost plateaued after that number of features, however, in the model selection step models trained on the reduced set of features consistently showed worse performance, ultimately leading to the full set of features to be used.  
Considering the nature of the features, that are all answers to a questionnaire, and therefore very easy to obtain, there is no strong need to reduce the number of features except in the case of performance gain. 
##### 2. Model selection:
Model selection process comprised of a number of substeps:  
Firstly, models were evaluated based on their MCC, recall and recall variance. Poor performers at this step (naive bayes methods and RUS Boost) were discarded.  
Secondly, the problem was transformed from a multiclass problem to a binary problem. Class 1 and class 2 were fused into class 1, representing unhealthy patients.  
Models were evaluated based on their MCC and recall on the new binary problem. Threshold tuning was performed to maximize recall of prediabetics especially. The logic behind this choice is that, in a real case scenario is more important to detect prediabetics instead of diabetics since it is more likely that a person is prediabetic and does not know it, instead of being diabetic and not knowing it, since diabetes is a morbid medical condition that seldom goes unnoticed by the patient, while the same is not true for prediabetes.  
At this step balanced random forest and easy ensemble were discarded.  
PARLARE DELL ULTIMO STEP DELL HIERARCHICAL APPROACH E SPIEGARNE IL FALLIMENTO 



| Comando | Descrizione | Note |
| --- | --- | --- |
| `git status` | Mostra lo stato dei file | Molto utile |
| `git push` | Carica le modifiche | Richiede auth |
| `git pull` | Scarica le modifiche | Aggiorna locale |

