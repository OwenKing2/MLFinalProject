import csv
import itertools
import warnings
import nltk
import pandas as pd
import sklearn
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings(action='ignore')


def simplified_preprocessing(filename):
    header = ["train_id", "Sentence_1", "Sentence_2", "Output"]
    df = pd.read_csv(filename, sep='\t', names=header, engine='python', encoding='utf8', error_bad_lines=False,
                     quoting=csv.QUOTE_NONE)
    # Make all words lowercase
    df['Sentence_1'] = df['Sentence_1'].str.lower()
    df['Sentence_2'] = df['Sentence_2'].str.lower()

    return df


def bleu_score(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"])
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []

    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        first_sentence = nltk.word_tokenize(sentence_1)
        second_sentence = nltk.word_tokenize(sentence_2)
        bleu1.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             weights=[1]))
        bleu2.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             weights=[0.5, 0.5]))
        bleu3.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             weights=[1 / 3, 1 / 3, 1 / 3]))
        bleu4.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             weights=[1 / 4, 1 / 4, 1 / 4, 1 / 4]))

    features["BLEU_1"] = bleu1
    features["BLEU_2"] = bleu2
    features["BLEU_3"] = bleu3
    features["BLEU_4"] = bleu4

    return features


def meteor_scores(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["METEOR"])

    meteor_score = []
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        first_sentence = nltk.word_tokenize(sentence_1)
        second_sentence = nltk.word_tokenize(sentence_2)
        meteor_score.append(nltk.translate.meteor_score.single_meteor_score(first_sentence, second_sentence))
    features["METEOR"] = meteor_score
    return features


def character_bigrams_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["CharacterBigramUnion", "CharacterBigramIntersection",
                                     "NumCharBigrams1", "NumCharBigrams2",
                                     ])
    bigramUnion = []
    bigramIntersection = []
    numbigrams1 = []
    numbigrams2 = []

    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        sentence_1_no_spaces = sentence_1.replace(" ", "")
        sentence_2_no_spaces = sentence_2.replace(" ", "")
        sentence_1_char_bigrams = [sentence_1_no_spaces[i:i + 2] for i in range(len(sentence_1_no_spaces) - 1)]
        sentence_2_char_bigrams = [sentence_2_no_spaces[i:i + 2] for i in range(len(sentence_2_no_spaces) - 1)]
        bigram_matches = 0
        for phrase in sentence_1_char_bigrams:
            if phrase in sentence_2_char_bigrams:
                bigram_matches += 1
        bigramIntersection.append(bigram_matches)
        bigramUnion.append(len(sentence_1_char_bigrams) + len(sentence_2_char_bigrams))
        numbigrams1.append(len(sentence_1_char_bigrams))
        numbigrams2.append(len(sentence_2_char_bigrams))
    features["CharacterBigramUnion"] = bigramUnion
    features["CharacterBigramIntersection"] = bigramIntersection
    features["NumCharBigrams1"] = numbigrams1
    features["NumCharBigrams2"] = numbigrams2

    return features


def word_unigram_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["SentenceUnigramUnion", "SentenceUnigramIntersection",
                                     "NumSentUnigrams1", "NumSentUnigrams2"])
    unigramUnion = []
    unigramIntersection = []
    numunigrams1 = []
    numunigrams2 = []

    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        sentence_1_words = nltk.word_tokenize(sentence_1)
        sentence_2_words = nltk.word_tokenize(sentence_2)
        sentence_1_unigrams = list(nltk.ngrams(sentence_1_words, 1))
        sentence_2_unigrams = list(nltk.ngrams(sentence_2_words, 1))
        unigram_matches = 0
        for phrase in sentence_1_unigrams:
            if phrase in sentence_2_unigrams:
                unigram_matches += 1
        unigramIntersection.append(unigram_matches)
        unigramUnion.append(len(sentence_1_unigrams) + len(sentence_2_unigrams))
        numunigrams1.append(len(sentence_1_unigrams))
        numunigrams2.append(len(sentence_2_unigrams))
    features["SentenceUnigramUnion"] = unigramUnion
    features["SentenceUnigramIntersection"] = unigramIntersection
    features["NumSentUnigrams1"] = numunigrams1
    features["NumSentUnigrams2"] = numunigrams2
    return features


def all_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=[
        "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4",
        "Meteor Score",
        "CharacterBigramUnion", "CharacterBigramIntersection", "NumCharBigrams1", "NumCharBigrams2",
        "SentenceUnigramUnion", "SentenceUnigramIntersection", "NumSentUnigrams1", "NumSentUnigrams2",
    ])
    bleu_scores = bleu_score(sentence1array, sentence2array)

    features["BLEU_1"] = bleu_scores["BLEU_1"]
    features["BLEU_2"] = bleu_scores["BLEU_2"]
    features["BLEU_3"] = bleu_scores["BLEU_3"]
    features["BLEU_4"] = bleu_scores["BLEU_4"]

    features["Meteor Score"] = meteor_scores(sentence1array, sentence2array)

    char_bigram = character_bigrams_features(sentence1array, sentence2array)
    word_unigram = word_unigram_features(sentence1array, sentence2array)

    features["CharacterBigramUnion"] = char_bigram["CharacterBigramUnion"]
    features["CharacterBigramIntersection"] = char_bigram["CharacterBigramIntersection"]
    features["NumCharBigrams1"] = char_bigram["NumCharBigrams1"]
    features["NumCharBigrams2"] = char_bigram["NumCharBigrams2"]

    features["SentenceUnigramUnion"] = word_unigram["SentenceUnigramUnion"]
    features["SentenceUnigramIntersection"] = word_unigram["SentenceUnigramIntersection"]
    features["NumSentUnigrams1"] = word_unigram["NumSentUnigrams1"]
    features["NumSentUnigrams2"] = word_unigram["NumSentUnigrams2"]

    return features


#
#
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)


# df_tensor = df_to_tensor(X)
# print(df_tensor)
training_data = simplified_preprocessing("train_with_label.txt")
X = all_features(training_data["Sentence_1"], training_data["Sentence_2"])
y = training_data["Output"]

dev_data = simplified_preprocessing("dev_with_label.txt")
Xdev = all_features(dev_data["Sentence_1"], dev_data["Sentence_2"])
ydev = dev_data["Output"]

n_samples, n_features = X.shape
print(n_samples)
print(n_features)

scaler = sklearn.preprocessing.StandardScaler()
X = scaler.fit_transform(X)
Xdev = scaler.fit_transform(Xdev)

X = torch.from_numpy(X.astype(np.float32))
Xdev = torch.from_numpy(Xdev.astype(np.float32))
y = df_to_tensor(y)
ydev = df_to_tensor(ydev)
# y = torch.from_numpy(y.astype(np.float32))
# ydev = torch.from_numpy(ydev.astype(np.float32))


y = y.view(y.shape[0], 1)
ydev = ydev.view(ydev.shape[0], 1)


class Logistic_Reg_model(torch.nn.Module):
    def __init__(self, no_input_features):
        super(Logistic_Reg_model, self).__init__()
        self.layer1 = torch.nn.Linear(no_input_features, 20)
        self.layer2 = torch.nn.Linear(20, 1)

    def forward(self, x):
        y_predicted = self.layer1(x)
        y_predicted = torch.sigmoid(self.layer2(y_predicted))
        return y_predicted


model = Logistic_Reg_model(n_features)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

number_of_epochs = 100
for epoch in range(number_of_epochs):
    y_prediction = model(X)
    loss = criterion(y_prediction, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 10 == 0:
        print('epoch:', epoch + 1, ',loss=', loss.item())

with torch.no_grad():
    y_pred = model(Xdev)
    y_pred_class = y_pred.round()
    accuracy = (y_pred_class.eq(ydev).sum()) / float(ydev.shape[0])
    print(accuracy.item())

# class LogisticRegression(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegression, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         outputs = self.linear(x)
#         return outputs

# class Feedforward(torch.nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Feedforward, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
#         self.relu = torch.nn.ReLU()
#         self.fc2 = torch.nn.Linear(self.hidden_size, 1)
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         hidden = self.fc1(x)
#         relu = self.relu(hidden)
#         output = self.fc2(relu)
#         output = self.sigmoid(output)
#         return output
