import os
import string
import math

import nltk
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))
    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    data = {}
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            with open(os.path.join(dirpath, filename), 'r', encoding="utf8") as f:
                data[filename] = f.read()

    return data


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by converting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    tokens = document.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(tokens.lower())
    tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words('english')]
    return tokens


def get_all_words(documents):
    all_words = set()

    for words in documents.values():
        all_words.update(words)

    return all_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    total_docs = len(documents)

    idfs = {}
    all_words = get_all_words(documents)

    for word in all_words:
        f = sum(word in documents[filename] for filename in documents)
        idfs[word] = math.log(total_docs / f)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    # Calculate tf-idf:
    tfids = {filename: 0 for filename in files}
    for word in query:
        tfid = 0
        for filename, text in files.items():
            if word in text:
                tfid = text.count(word) * idfs[word]
            tfids[filename] += tfid

    # Sort files by tfid score
    tfids_sorted = {filename: tfid for filename, tfid in
                    sorted(tfids.items(), key=lambda item: item[1], reverse=True)}

    top_n = [name for name in tfids_sorted.keys()][:n]
    return top_n


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    ranking = []
    for sentence, words in sentences.items():
        idf = 0
        matches = 0
        for word in query:
            if word in words:
                matches += 1
                idf += idfs[word]

        density = matches / len(words)
        ranking.append((sentence, idf, density))

    # Sort by idf and term density
    ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [entry[0] for entry in ranking[:n]]


if __name__ == "__main__":
    main()
