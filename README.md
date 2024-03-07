# **Hotel Review Classification**

### ***Problem Formulation***
* The given problem is a binary classification problem.The task is to classify hotel reviews as truthful or deceptive.As naive bayes being the popular 
  classification algorithm to classify spam messages, the same has been used.
* In the naive bayes classification, we calculate posterior probability for each label i.e. truthful and deceptive given the review.
*  Naive bayes is based on bayes theorem which is given by 
*      P(A | B) =( P(B | A) * P(A) ) / P(B) 
* Here we try to calculate probability of occurrence  of an event A given an observed event B
*  In the similar way,according to naive bayes algorithm, we calculate probability of the review(w1,w2,..,wn) being truthful as below:
*      P (truthful | w1,w2,..wn) = (P(w1 | truthful)*P(w2 | truthful)..P(wn | truthful)) * P(truthful) 
* We neglect the denominator which is P(w1,w2,...wn) as it's a constant.
* In the similar way,we calculate probability of the review(w1,w2,..,wn) being deceptive as below:
*      P (deceptive| w1,w2,..wn) = (P(w1 | deceptive)*P(w2 | deceptive)..P(wn | deceptive)) * P(deceptive).
* Here P (truthful) is the probability of review being truthful and P(deceptive) is the probability of review being deceptive as below:
*      P(truthful) = count of truthful reviews / total number of reviews
*      P(deceptive) = count of deceptive reviews / total number of reviews
* While training the model,we consider the review texts as a bag of words and try to calculate truthful  likelihood probabilities for all the words given the labels as below:
*      P( w | truthful) = count of w occuring in truthful reviews / total count of truthful words
* Similarly, we calculate the deceptive likelihood probabilities for all the words given the label as below:
*      P( w | deceptive) = count of w occuring in deceptivereviews  /  total count of deceptive words
* During testing, we calculate the truthful and deceptive probabilities of reviews which are P (truthful | w1,w2,..wn) and P (deceptive| w1,w2,..wn) as mentioned above
* By comparing the above calculated probabilities, we classify the review as truthful or deceptive.

### ***Program Description***
* We first find totalTruthProbab which is P(truthful) and totalDecepProbab which is P(deceptive) as per the above formula
* We train the model using deceptive.train.txt.For each review, we remove punctuations in the review string and convert it to lower case. Also, we ignore stop words.These are the only preprocessing steps done.
* Next we split the review using space as delimiter, and now we have the bag of words.
* We calculate the count for each word being truthful or deceptive based on the label and store it in the truthWordCountMap or decepWordCountMap respectively.
* We add the missing words in both the truthWordCountMap  and decepWordCountMap and initialise the count to zero
* We assume alpha value to 1 and increment count for each word in both truthWordCountMap and decepWordCountMap by 1.
* We calculate the likelihood of words by iterating through truthWordCountMap and decepWordCountMap and store them in wordGivenTruthProbabMap or wordGivenDecepProbabMap respectively.
* Next we test data using deceptive.test.txt.For each review, same preprocessing will be done to match the words which we have already got from training data.
* P (truthful | review(w1,w2,..wn)) and P (deceptive| review(w1,w2,..wn)) will be calculated for each review which are truthfulProbability and deceptiveProbability.
* If P (truthful | review(w1,w2,..wn)) is greater than or equal to P (deceptive| review(w1,w2,..wn)), we classify the review as truthful else we classify it as deceptive

### ***Assumptions and Experiments and Observations***
* As part of preprocessing, we remove all the punctuations from the review.This is done to minimise the number of missing words bewteen training and testing data set.
* If punctuations are not removed, we might miss including the likelihood of that word because of the punctuation.
* While testing, we observed accuracy of model increased by 2.75% when preprocessing step was removed for testing data, accuracy increased. At this step, we put missing words into a file and observed that several known word
probabilities were not included because of the punctuations.
* We used all the stop words from nltk library because the count of these are very high and model focuses on useful information
* To increase the accuracy of the naive bayes model, initially we used alpha as 1 for the missing words between training and testing data.This proved to nbe effective and increased accuracy to a good extent.
* We used both stemming and lemmatization techniques to increase accuracy.For the given data, these techniques didn't push the accuracy up, rather there was a decrease in accuracy by 0.5-2%. 
