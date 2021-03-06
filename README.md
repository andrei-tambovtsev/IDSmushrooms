# IDSmushrooms
# Repository for Introduction to Data Science (LTAT.02.002) project 2020.  
## Authors Andrei Tambovtsev and Semjon Kravtšenko

### Overview
Repository contains files needed for Data Science project
Project idea is to train a machine learning model to help novice mushroom hunters models and (possibly) finding some interesting correlations for the same goal.
We used mushrooms.cvs dataset from kaggle (https://www.kaggle.com/uciml/mushroom-classification)
Original dataset included in repository main branch

### B6_report
B6_report.docx and .pdf is text file describing first stage of project: Buisnes undrestanding, Data undrestanding and Plan of project. (B6 reffers to project ID) 

### KAGGLE-EDIBILITY-OF-MUSHROOMS
KAGGLE-EDIBILITY-OF-MUSHROOMS.ipynb is jupyter notebook file, that contains commented tests perfomed on dataset using python3 and some external libraries (mostly 'pandas' and 'sklearn') This notebook step by step shows our approach and way of thinking. 

### Script
script.py is python3 file, that contains code to train K-nearest-neighbors and RandomForestClassifier algorithms from sklearn library and obsereve their prediction score. Script was also included in KAGGLE-EDIBILITY-OF-MUSHROOMS.ipynb with additional comments.
It also creates a trained model to be used in the pygame program.

### Program (pygame)
Proof of concept of a software, that helps novice mushroom hunters. Made to be reasonably easy to modify, if we decide to change criteria used.

### Trees
trees folder contains .pdf files with visualization on decision tree classifier, implemented on mushrooms.cvs dataset. Each .pdf file contains decision tree with different attribute sets

#### folder
pygame folder contains programm that was final aim of a project. simple classification code that can predict edability of mushroom based on 3 easy recognizable attributes of mushroom (stalk shape, stalk root and gill color)

#### Imgs
pygame folder contains also folder with images, that _main_.py is using for interface.

#### _main_
code in _main_.py is written on python3, using time library and 5 additional sets of modules that are need to be installed using commands: "pip install pygame", "pip install pygame-widgets", "pip install pickle-mixin", "pip install sklearn" and "pip install numpy". Aftr installing additional sets coda can be run and it must show graphic interface with questions about mushroom. After answering 3 questions interface show 1 of 2 images that predicts whether mushroom edible or not.

### Poster
B6_poster.pdf is presentation poster with information about project, our aims, methods and acievments. Poster was made using https://lucid.app/ and converted into .pdf format. Poster also contains images of _main_.py and visualization of mushroom parts (from publicly available mushroom tutorial slides https://pt.slideshare.net/rayborg/mushroom-tutorial/) that we choose for our model.

### Conclusion
We hope that our project satisfies all needed criteria. We definitly gained knowledge while working on it. Especially interesting was using regression functions "not properly" (in this bit we went "creative" on how the use tools we learned in this course), to find strongest corellation, encoding 23 columns on dataset into 114 and then reducing used data to 3 most valuable parameters.
