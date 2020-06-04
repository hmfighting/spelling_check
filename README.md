# To facilitate communication with relevant researchers, both code and data have been uploaded. 
Among them, there are two datasets to validate the model. One is the SIGHAN Bake-off dataset, the other is the logistics dataset. 
Since the logistics dataset is a private dataset provided by the logistics company, this part of data has not been uploaded. 
The main_word2vecPublic.py is utilized to train and test the model. 
And the process_word2vecPublic is used to calculate the decision weight matrix and to process the data.
The word_vector file contains code to train the word vector model. 

Steps to use the above file:
step1: Pre-trained word2vec model. This part of the code comes from word_vector directory
step2: Build decision weight and prepare input data for the model according to process_word2vecPublic.py
step3: Train and test model by main_word2vecPublic.py




The Baselines reference https://github.com/shibing624/pycorrector
