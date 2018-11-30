import gzip
import gensim
import logging
import random
import os
import sys

"""
Word2Vec Playground
--- 
Loosely based off of the tutorial here.
http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code
---
On my system I've got the entirety of English Wikipedia attached to this so watch out.
"""

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load():
    """
    Using a generator so your computer doesn't melt!
    Was kind of hoping that I could make this have a littler loading bar.
    STDOUT seems to be locked or the system is having it's resources hogged.
    """
    print("These files are available to use as a corpus:")
    
    file_list = []
    for name in os.listdir("data"):
        if name.split(".")[-1] == "gz":
            file_list.append(name)

    file_select = {}
    for file in file_list:
        short = file.split(".")[0]
        file_select[short] = file
        print("\t" + short)

    filename = file_select[input("Choice: ")]
    logging.info("Loading and decompressing the " + filename + " file")
    print("Loading", end="", flush=True)
    rev_count = 0
    for _,v in enumerate(gzip.open("data/" + filename)):
        rev_count += 1
        if rev_count % 5000 == 0:
            print(end=".", flush=True)
        yield gensim.utils.simple_preprocess(v)
    print(flush=True)
    logging.info("Loaded " + str(rev_count) + " reviews.")

def train(data):
    """
    Create the model. Pretty straightforward due to the API.
    """
    logging.info("Generating model")
    model = gensim.models.Word2Vec(
        [x for x in data],
        size=150,
        window=2,
        min_count=2,
        workers=10
    )
    logging.info("Training model.")
    model.train(data, total_examples=len(list(data)), epochs=10)
    return model

def play():
    """
    The Playground.

    Some lovely functions and a basic interface so you can have
    a bit of fun.
    """

    # Setup the model
    model = train(load())

    # Determine which words could be used (statistically) in place of a chosen word.
    def similar():
        word = gensim.utils.simple_preprocess(
                input("Enter the word you would like to find similar words for: ")
            )[0]
        top = int(input("How many words do you want to find?: "))
        result = ""
        sim_list = model.wv.most_similar(positive=word, topn=top)
        result += "The most similar words to " + word + " are:\n"
        for item in sim_list:
            result += "\t" + item[0] + ": " + str(item[1] * 100) + "%\n"
        return result
    
    # Determine how similar words are based on the corpus (distance)
    def compare():
        word = gensim.utils.simple_preprocess(input("What is the first word you want to compare?: "))[0]
        other = gensim.utils.simple_preprocess(input("And the other one?: "))[0]
        return word + " is " + str(model.wv.similarity(w1=word, w2=other) * 100) + "% similar to " + other + " according to this model.\n"

    # Similar to similar() but used against a sentence with the benefit of random chance.
    def scramble():
        scrambled = []
        sentence = input("Enter your sentence to scramble.\nS: ")
        scrambliness = int(input("How many orders of dissimilarity are we allowed to use?\n[How many points away from the words]?: "))
        for word in gensim.utils.simple_preprocess(sentence):
            choice = random.randint(0, scrambliness - 1)
            chosen = model.wv.most_similar(positive=word, topn=scrambliness)[choice]
            scrambled.append(chosen[0])
        
        result = ""
        for word in scrambled:
            result += word + " "
        
        return "Your scrambled output is: \n" + result + "\n"

    # Provides instructions on how to use the UI
    def instruct():
        return """
\nWelcome to the Word2Vec Playground!
---

        Valid commands are:
        similar, compare, scramble, switch, help and quit.

        Have fun!
        """

    # Switch the corpus that is in use
    def switch_corpus():
        nonlocal model
        model = train(load())

        return "Successfully switched model!\n" + instruct() 

    # Closes the program
    def close():
        print("Thanks for giving this a go!")
        raise SystemExit

    # Called on a typo
    def bad_command():
        return "Invalid command! Type help for valid commands\n"

    # The command dictionary/lookup table
    commands = {
        "similar": similar,
        "compare": compare,
        "scramble": scramble,
        "switch": switch_corpus,
        "help": instruct,
        "quit": close
    }

    print(instruct())

    while True:
        f = commands.get(input("\nPlease enter a command: "), bad_command)
        # I guess I'm a little lazy with my error handling
        try:
            print(f())
        except SystemExit:
            raise SystemExit
        except:
            print("Bad input!\n")

play()