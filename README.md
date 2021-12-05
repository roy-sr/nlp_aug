# Problem statement  

# NLP Project: Data Augmentation

This archive contains a data file called labeled_messages.csv. labeled_messages.csv has ~3,500 user messages and empathy labels (the sensation that Woebot might empathize with) for each one.

We would like to have more examples to train a more robust classifier. Please program a data augmentation algorithm that outputs variations of the messages without altering their semantics with respect to their labels. Note that some rows are labeled with multiple classes. If possible, please also provide or describe an automated heuristic to verify that the original and augmented examples are semantically invariant within reason.

In addition to the programming task, please also provide some analysis to highlight key resulting improvements (ie, number of new examples, syntantic variations, etc). Please identify difficult examples and offer thoughts on why your technique doesn't work as well on them. Feel free to take into account issues like class balance if appropriate.

Please commit relevant code and description to a git repo and submit it all as a compressed file. And be prepared to brainstorm ways to improve your implementation :)


# Requirements
a. Python NLTK lib

b. Tensorflow

c. Tensorflow hub

d. Scipy

e. Pandas

f. Paraphrase database http://paraphrase.org/#/download

# Step 1
Prepare the paraphrase binary file

a. Download and extract 'ppdb-2.0-tldr.gz' file from http://paraphrase.org/#/download and extract it into dataset folder

b. Execute prepare_ppdb_dict.py. It will create 'ppdb_model.bin' file in the dataset folder. This file is the extraction of the required fields from the PPDB Dataset.

# Step 2
Create augmented dataset

a. Execute aug.py to generate the augmented dataset 'new_labeled_messages.csv'


# Similarity Heuristic 

Semantic similarity between two phrase can be calculated based on the following points:

a. Knowledge-Based Similarity i.e concept by a node in an ontology graph.

b. Language Model-Based Similarity i.e.

    Removing stop words
    Tagging the two phrases using any Part of Speech (POS) algorithm
    From the tagging step output, this type forms a structure tree for each phrase (parsing tree)
    Building undirected weighted graph using the parsing tree
    Finally, the similarity is calculated as the minimum distance path between nodes (words)

c. Statistical-Based Similarity i.e Vectors representation techniques.

In the present task I used the 'Statistical-Based Similarity' heuristic to check the semantically invariant phrases. I converted the both input and output message into 512 embedding vectors using Universal Sentence Encoder (Transformer based) and evaluated the 'cosine similarity' between them (considering values with gt 0.75 as similar phrases). The USE embeddings takes care of both lexical and contexctual meaning of the tokens.


# Rooms for Imporvement
a. Inclusion of Emoji for better understanding of the context.
b. Rectify class imbalance specially for multi-label taret by including more datapoints
c. Inclusion of english slang e.g. high means intoxicated, numb means disconnected
d. Better formation of sentences e.g (source) i am getting better at controlling my emotions --> (output) i am getting improvements at controlling my emotions --> (improvements) i am improving controlling my emotions 




