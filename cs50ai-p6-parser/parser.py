import string

import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to" | "until"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | S Conj VP | PP NP | P NP Adv | S P NP
NP -> N | Det N | PP NP | NP PP | VP NP | NP VP | Det AdjP N
VP -> V | V PP | Adv V | V Adv | V NP | Adv V NP | NP VP Adv | PP V
PP -> P NP | P
AdjP -> Adj | AdjP Adj
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    tokens = nltk.word_tokenize(sentence.lower())
    tokens = [token for token in tokens if is_valid_token(token)]

    return tokens


def is_valid_token(token: str) -> bool:
    """
    Checks if a token contains at least one alphabetical character.
    :param token: token
    :return: True, if the token contains at least one alphabetical character.
    """
    for c in token:
        if c in string.ascii_lowercase:
            return True
    return False


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """

    np_chunks = []
    for subtree in tree.subtrees(lambda st: st.label() == 'NP'):
        if not contains_np(subtree):
            np_chunks.append(subtree)

    return np_chunks


def contains_np(subtree):
    np_subtrees = [sub for sub in subtree.subtrees(lambda st: st.label() == 'NP' and st != subtree)]
    return len(np_subtrees) > 0


if __name__ == "__main__":
    main()
