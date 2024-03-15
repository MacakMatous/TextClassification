#!/usr/bin/python
#
# Klasifikace textu pomocí umělé neuronové sítě
# Ukázka pro 4IZ232 podle Kotzias et. al (2015) a Zamazal (2020)
#


###
# Načtení datové sady
###
import pandas

imdb_dataset = pandas.read_csv('imdb_labelled.csv', names=['sentence', 'label'], sep='\t')

sentences = imdb_dataset['sentence'].values
labels = imdb_dataset['label'].values

# kontrolní výpis
print(sentences.shape)
print(labels.shape)
print("{}:{}".format(sentences[10],labels[10]))


###
# Rozdělení na trénovací a testovací množinu
###
from sklearn.model_selection import train_test_split

sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=0.25, random_state=1000)

# kontrolní výpis
print("Rozdělená data:")
print("trénovací data:")
print(sentences_train.shape)
print(labels_train.shape)
print("{}:{}".format(sentences_train[9],labels_train[9]))
print("testovací data:")
print(sentences_test.shape)
print(labels_test.shape)
print("{}:{}".format(sentences_test[9],labels_test[9]))


###
# Předzpracování textu do tvaru Bag of Words
###
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary = True)
vectorizer.fit(sentences_train)

# kontrolní výpis (kódy slov)
print(vectorizer.vocabulary_['and'])
print(vectorizer.vocabulary_['jimmy'])
print(vectorizer.vocabulary_['movie'])
print(sentences_train[9])

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

# kontrolní výpis (zakódované věty)
print(X_train.shape)
print(X_train[9,:])
print(X_test.shape)
print(X_test[9,:])

y_train = labels_train
y_test = labels_test


###
# Umělá neuronová síť
# vícevrstvý perceptron
###

from keras.models import Sequential
from keras import layers


activation_functions = ['relu',
                       'sigmoid', 
                       'tanh',]

optimizer_functions = ['sgd', 
              'adam',
              'RMSprop',
              ]

loss_functions = ['binary_crossentropy',
                  'mean_squared_error',
                  'hinge']

for a in activation_functions:
    for o in optimizer_functions:
        for l in loss_functions:
            # počet atributů
            input_dim = X_train.shape[1]
            print(input_dim)

            # model neuronové sítě
            model = Sequential()

            model.add(layers.Dense(5, input_dim=input_dim, activation=a))
            model.add(layers.Dense(1, activation=a))
            model.compile(loss=l, optimizer=o, metrics=['accuracy'])
            model.summary()

            ###
            # Umělá neuronová síť
            # základní učení a vyhodnocení
            ###

            # natrénování sítě
            history = model.fit(X_train, y_train, epochs=60, verbose=True, batch_size=10)

            # vyhodnocení sítě
            loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

            print("Správnost modelu na trénovacích datech: {:.4f}".format(accuracy))

            loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

            print("Správnost modelu na testovacích datech:  {:.4f}".format(accuracy))


            ###
            # Vizualizace vyhodnocení
            ###

            from matplotlib import pyplot

            print(history.history.keys())
            accuracy_train = history.history['accuracy']
            loss_train = history.history['loss']

            epochs_accuracy = range(1, len(accuracy_train) + 1)
            epochs_loss = range(1, len(loss_train) + 1)

            pyplot.plot(epochs_loss, loss_train, 'bo', label='Ztráta při trénování')
            pyplot.title('Průběh ztrátové funkce během trénování')
            pyplot.xlabel('Epocha')
            pyplot.ylabel('Hodnota ztrátové funkce')
            pyplot.legend()

            pyplot.savefig(f'output/{a}-{o}-{l}.png')
            #pyplot.show()

            pyplot.clf()   # vyčištění obrázku

            pyplot.plot(epochs_accuracy, accuracy_train, 'bo', label='Úspěšnost při trénování')
            pyplot.title('Průběh úspěšnosti během trénování')
            pyplot.xlabel('Epocha')
            pyplot.ylabel('Úspěšnost')
            pyplot.legend()

            pyplot.savefig(f'output/{a}-{o}-{l}.png')
            #pyplot.show()
            pyplot.clf()   # vyčištění obrázku


            ###
            # Uložení a načtení modelu
            ###

            from keras.models import load_model

            model.save(f'output/{a}-{o}-{l}-mlp.h5')
            del model
            model = load_model(f'output/{a}-{o}-{l}-mlp.h5')
            model.summary()

