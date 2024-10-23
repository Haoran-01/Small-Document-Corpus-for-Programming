import argparse
import math
import os
import re
import time
import files.porter as porter


# Manual input search
def interactive():
    # Read documents and stopword
    script_dir = os.path.dirname(os.path.abspath(__file__))
    documents_path = os.path.join(script_dir, "documents_2")
    stopwords_path = os.path.join(script_dir, "files", "stopwords.txt")

    documents = read_documents_info(documents_path)
    stopwords = read_stopword_file(stopwords_path)
    p = porter.PorterStemmer()

    # Create or read index
    index, documents, avg_doclen = create_index(documents, stopwords, p)

    # Read query and perform search
    while True:
        query = input("Enter a query (or 'QUIT' to exit): ")
        if query == "QUIT":
            break
        else:
            # Search using the bm25 model
            results = bm25_model(query, documents, index, 1, 0.75, avg_doclen)[:15]
            rank = 1
            for result in results:
                print(str(rank) + " " + result[0] + " " + str(result[1]))
                rank += 1


# Automatic search
def automatic():
    # Read documents and stopword
    load_start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    documents_path = os.path.join(script_dir, "documents_2")
    stopwords_path = os.path.join(script_dir, "files", "stopwords.txt")

    documents = read_documents_info(documents_path)
    stopwords = read_stopword_file(stopwords_path)
    p = porter.PorterStemmer()

    # Create or load index
    index, documents, avg_doclen = create_index(documents, stopwords, p)
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    print(f"Program load time：{load_time} seconds")

    print("Loading end")
    start_time = time.time()
    # Open the "queries.txt" file to read the query
    with open("files/queries.txt", "r") as queries_file:
        queries = queries_file.readlines()

    # Open the "results.txt" file to write the results
    with open("files/results.txt", "w") as results_file:
        # Iterate through each query
        for query in queries:
            query_terms = query.strip().split(" ")
            query_id = query_terms[0]
            query_text = clear_txt(" ".join(query_terms[1:]).strip(), stopwords, p)

            # Calculate the similarity score between the query and the document to get a ranked list
            ranking = bm25_model(query_text, documents, index, 1, 0.75, avg_doclen)
            rank_number = 1

            # Iterate through the list of rankings and write the results to the "results.txt" file
            for rank in ranking:
                results_file.write(f"{query_id}\t{rank_number}\t{rank[0]}\t{str(rank[1])}\n")
                rank_number += 1

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Program search time：{runtime} seconds")
    return


# Create index on first run, load index later
def create_index(documents, stopwords, p):
    # Read the index
    index_file_path = "index.txt"
    if os.path.exists(index_file_path):
        processed_doc = {}
        # If the index file exists, load the index
        index, avg_doclen = load_index(index_file_path)
        # Processing of articles
        for doc_id, document in documents.items():
            processed_doc[doc_id] = clear_txt(documents[doc_id], stopwords, p)
    else:
        # If index file does not exist, create index and save to file
        index = {}
        processed_doc = {}
        for doc_id, document in documents.items():
            # Clear punctuation
            document = clear_pun(document)
            words = document.split()
            clean_words = set()
            all_words = []

            for word in words:
                if word not in stopwords:
                    # stemming
                    stem_word = p.stem(word)
                    all_words.append(stem_word)  # Storage of processed words
                    # Create index, index structure is {term : {doc_id : [doc_1, doc_2], idf : idf_value}}
                    if stem_word not in clean_words:
                        clean_words.add(stem_word)
                        if stem_word in index:
                            index[stem_word]["doc_id"].add(int(doc_id))
                        else:
                            index[stem_word] = {}
                            index[stem_word]["doc_id"] = set()
                            index[stem_word]["doc_id"].add(int(doc_id))
            processed_doc[doc_id] = all_words
        # Average length of stored documents
        avg_doclen = calculate_avg_doc_len(processed_doc)

        # Calculate document idf
        document_numbers = len(processed_doc)
        for term in index:
            df = len(index[term]["doc_id"])
            idf = math.log((document_numbers - df + 0.5) / (df + 0.5))
            index[term]["idf"] = idf

        # Storage data
        save_index(index, "index.txt", avg_doclen)
    return index, processed_doc, avg_doclen


# Clear punctuation and numbers
def clear_pun(document_content):
    clean_document = re.sub(r"[^\w\s]|[\d]", '', document_content)
    return clean_document


# Process text, remove punctuation, numbers, and stemming
def clear_txt(txt, stopwords, p):
    clean_txt = re.sub(r"[^\w\s]|[\d]", '', txt)
    words = clean_txt.split()
    clean_words = []

    for word in words:
        if word not in stopwords:
            # stemming
            word = p.stem(word)
            clean_words.append(word)
    return clean_words


# Creating an index
def save_index(index, file_path, avg_doclean):
    the_key = "avg_doclean"
    with open(file_path, 'w') as file:
        file.write(f"{the_key}: {avg_doclean}\n")
        for key, value in index.items():
            str_idf = value["idf"]
            str_doc_id = ", ".join(str(item) for item in value["doc_id"])
            file.write(f"{key}: {str_idf}")
            file.write(f", {str_doc_id}\n")


# Load the index
def load_index(file_path):
    print("Loading BM25 index from file, please wait.")
    index = {}
    with open(file_path, 'r') as file:
        avg_length = file.readline()
        lines = file.readlines()
        avg_length = avg_length.strip().split(': ')[1]
        for line in lines:
            key, value = line.strip().split(': ')
            index[key] = value.split(', ')
    return index, avg_length


# Read stored documents
def read_documents_info(folder_path):
    document_collection = {}  # Hash tables for storing collections of documents

    # Iterate through all the files in the folder
    for file_id in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_id)

        # Read the contents of a file
        with open(file_path, 'r', encoding='utf-8') as file:
            # Convert all text to lower case
            document_content = file.read().lower()
            # Adding document content to a document collection
            if len(document_content) != 0:
                document_collection[file_id] = document_content

    return document_collection


# Read stopword files
def read_stopword_file(stopwords_file_path):
    with open(stopwords_file_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords


# BM25 model
def bm25_model(query, documents, index, k, b, avg_doclen):
    scores = {}
    for term in query:
        if term in index:
            # The first run has a different index structure than subsequent runs, get idf
            if type(index[term]) is list:
                idf = float(index[term][0])
            else:
                idf = float(index[term]["idf"])
            for doc_id, doc_content in documents.items():
                # Calculation of tf and BM25
                tf = doc_content.count(term)
                score = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * (len(doc_content) / float(avg_doclen))))
                if doc_id in scores.keys():
                    scores[doc_id] += score
                else:
                    scores[doc_id] = score
    sorted_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_documents


# Calculate average document length
def calculate_avg_doc_len(documents_clean):
    documents_numbers = len(documents_clean)
    total_word_numbers = 0
    for doc_id, doc_content in documents_clean.items():
        word_numbers = len(doc_content)
        total_word_numbers += word_numbers
    avg_doclen = total_word_numbers / documents_numbers
    return avg_doclen


def main():
    parser = argparse.ArgumentParser(description='Small Corpus Search Program')
    parser.add_argument('-m', '--mode', choices=['automatic', 'interactive'], required=True,
                        help='Specify the mode to run the program')
    args = parser.parse_args()

    if args.mode == 'automatic':
        automatic()
    elif args.mode == 'interactive':
        interactive()


if __name__ == '__main__':
    main()
