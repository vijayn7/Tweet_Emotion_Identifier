import pandas as pd
from collections import defaultdict
from math import log

emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

class Classifier:
    def __init__(self):
        self.number_of_posts = 0
        self.unique_words = set()
        self.posts_per_word = defaultdict(int)
        self.posts_per_label = defaultdict(int)
        self.posts_per_label_word = defaultdict(lambda: defaultdict(int))
        self.log_prior = {}
        self.log_likelihood = defaultdict(lambda: defaultdict(float))
        self.predictions = []

    def getNextLine(self, df, linenumber):
        line = df.iloc[linenumber]
        return line
    
    def contentAsList(self, line):
        content = line['text']
        words_list = content.split()
        return words_list
    
    def getLabel(self, line):
        return line['label']
    
    def train(self, df, training_data_end):
        for i in range(training_data_end):
            self.number_of_posts += 1

            line = self.getNextLine(df, i)
            words_list = self.contentAsList(line)
            label = emotion_map[self.getLabel(line)]

            self.posts_per_label[label] += 1

            for word in words_list:
                self.unique_words.add(word)
                self.posts_per_word[word] += 1
                self.posts_per_label_word[label][word] += 1

        self.compute_log_prior()

        self.compute_log_likelihood()

        print("Trained on ", self.number_of_posts, " posts")

    def compute_log_prior(self):
        for label in emotion_map.values():
            self.log_prior[label] = log(self.posts_per_label[label].__float__() / self.number_of_posts.__float__())
    
    def compute_log_likelihood(self):
        for label in self.posts_per_label_word.keys():
            for word in self.posts_per_label_word[label].keys():
                self.log_likelihood[label][word] = log((self.posts_per_label_word[label][word].__float__()) / (self.posts_per_label[label].__float__()))

    def predict(self, df, training_data_end, test_data_end):
        for i in range(training_data_end, test_data_end):
            line = self.getNextLine(df, i)
            words_list = self.contentAsList(line)
            logProbabilities = self.getLogProbabilities(words_list)
            prediction = max(logProbabilities, key=logProbabilities.get)
            self.predictions.append(prediction)
        
    def word_does_not_exist(self, word):
        return word not in self.unique_words
    
    def word_does_not_exist_in_label(self, label, word):
        return word not in self.posts_per_label_word[label]

    def get_log_likelihood(self, label, word):
        if (self.word_does_not_exist(word)):
            return log(1.0 / self.number_of_posts.__float__())
        elif (self.word_does_not_exist_in_label(label, word)):
            total_occurences = 0
            for label in self.posts_per_label_word.keys():
                total_occurences += self.posts_per_label_word[label][word]
            return log(total_occurences.__float__() / self.number_of_posts.__float__())
        else:
            return self.log_likelihood[label][word]

    def getLogProbabilities(self, words_in_post):
        dict = {}
        for label in emotion_map.values():
            dict[label] = self.log_prior[label]
            for word in words_in_post:
                dict[label] += self.get_log_likelihood(label, word)

        return dict
    
    def print_results(self, df, training_data_end, test_data_end):
        correct_predictions = 0
        for i in range(training_data_end, test_data_end):
            line = self.getNextLine(df, i)
            # print("Predicted: ", self.predictions[i - training_data_end], " Actual: ", emotion_map[self.getLabel(line)])
            if self.predictions[i - training_data_end] == emotion_map[self.getLabel(line)]:
                correct_predictions += 1

        print("Accuracy: ", (correct_predictions / (test_data_end - training_data_end)) * 100, "%")

def main():
    df = pd.read_parquet('training_data.parquet')
    training_data_end = (len(df) * 0.75).__int__()
    test_data_end = len(df)
    # training_data_end = 30000
    # test_data_end = 30010

    classifier = Classifier()
    classifier.train(df, training_data_end)
    classifier.predict(df, training_data_end, test_data_end)
    classifier.print_results(df, training_data_end, test_data_end)

if __name__ == "__main__":
    main()