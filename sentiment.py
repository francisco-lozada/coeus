import os
from sklearn.feature_extraction.text import CountVectorizer     # machine learning library
from sklearn.ensemble import RandomForestClassifier     # machine learning library
import Word2VecUtil
import pandas as pd     # work with csv file
import nltk     # remove stop words

if __name__ == '__main__':
    # Read the data
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)
    
    input('Press Enter to continue...')

    # Clean the training data

    #print('Downloading stop words from the natural language tool kit...')
    #nltk.download()
    # Download directory: C:\Users\Franco\AppData\Roaming\nltk_data

    clean_train_reviews = []
    print('Cleaning and parsing the training set movie reviews...')
    for i in xrange(0,len(train["review"])):
        clean_train_reviews.append(" ".join(
            Word2VecUtil.review_to_wordlist(train["review"][i], True))
        )
    
    # Tokenization of words
    print('Tokenizing the words...\n')
    vectorizer = CountVectorizer(analyzer = "word", 
                                tokenizer = None, 
                                preprocessor = None, 
                                stop_words = None, 
                                max_features = 5000) 

    # Fit transform model to bag of words and create the feature vectors
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    # Train the classifier which consistes of a random forest with n number of decision trees
    print('Training the random forest (this may take a while)...')
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit(train_data_features, train["sentiment"])
    clean_test_reviews = []

    # Format the testing data
    print("Cleaning and parsing the test set movie reviews...\n")
    for i in xrange(0, len(test["reviews"])):
        clean_test_reviews.append(" ".join(
            Word2VecUtil.review_to_wordlist(train["review"][i], True))
        )
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()


    # Predict the classification of the reviews in the testing data
    print('Predicting test labels...')
    result = forest.predict(test_data_features)
    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
    output.to_csv(os.path.join(os.path.dirname(__file__), 
    'data', 'sentimentPredictions.csv'), index=False, quoting=3)
    print('Wrote results to sentimentPredictions.csv')
    