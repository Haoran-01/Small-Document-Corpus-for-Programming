import math
import os
import re
import time
import json
import files.porter as porter
from collections import defaultdict


def interactive():
    # 读取文档以及stopword
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_path = os.path.join(script_dir, "files", "stopwords.txt")
    documents_folder = "documents2"

    stopwords = read_stopword_file(stopwords_path)
    p = porter.PorterStemmer()

    index, documents, avg_doclen = create_index(documents_folder)
    while True:
        query = input("Enter a query (or 'QUIT' to exit): ")
        if query == "QUIT":
            break
        else:
            results = bm25_model(query, documents, index, 1, 0.75, avg_doclen)
            rank = 1
            for result in results:
                print(str(rank) + " " + result[0] + " " + str(result[1]))
                rank += 1


def automatic():
    # 读取文档以及stopword
    load_start_time = time.time()
    documents_folder = "documents"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_path = os.path.join(script_dir, "files", "stopwords.txt")

    print("load stopwords")
    stopwords = read_stopword_file(stopwords_path)
    p = porter.PorterStemmer()
    print("load stopwords end")

    index, documents, avg_doclen, tf_dict, len_dict = create_index(documents_folder)
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    print(f"程序加载时间：{load_time}秒")

    print("Loading end")
    start_time = time.time()
    # 打开 "queries.txt" 文件以读取查询
    with open("files/queries.txt", "r") as queries_file:
        queries = queries_file.readlines()

    # 打开 "results.txt" 文件以写入结果
    with open("files/results.txt", "w") as results_file:
        # 遍历每个查询
        for query in queries:
            query_terms = query.strip().split(" ")
            query_id = query_terms[0]
            query_text = clear_txt(" ".join(query_terms[1:]).strip(), stopwords, p)

            # 计算查询与文档的相似度分数，得到排名列表
            ranking = bm25_model(query_text, index, 1, 0.75, avg_doclen, tf_dict, len_dict)
            rank_number = 1

            # 遍历排名列表，写入结果到 "results.txt" 文件
            for rank in ranking:
                if rank[1] > 0:
                    results_file.write(f"{query_id}\t{rank_number}\t{rank[0]}\t{str(rank[1])}\n")
                    rank_number += 1
    end_time = time.time()
    runtime = end_time - start_time
    print(f"程序运行时间：{runtime}秒")
    return


def create_index(documents_folder):
    # 读取索引
    index_file_path = "index.json"
    if os.path.exists(index_file_path):
        processed_doc = {}
        print("have index")
        # 如果索引文件存在，加载索引
        index, avg_doclen, tf_dict, len_dict = load_index(index_file_path)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stopwords_path = os.path.join(script_dir, "files", "stopwords.txt")

        documents = read_documents_info(documents_folder)
        stopwords = read_stopword_file(stopwords_path)
        p = porter.PorterStemmer()
        # 如果索引文件不存在，创建索引并保存到文件
        index = {}
        processed_doc = {}
        tf_dict = {}
        len_dict = {}
        for doc_id, document in documents.items():
            # 清除标点符号
            document = clear_pun(document)
            words = document.split()
            clean_words = set()
            all_words = []
            term_fre = defaultdict(int)
            for word in words:
                if word not in stopwords:
                    # stemming
                    stem_word = p.stem(word)
                    all_words.append(stem_word)
                    term_fre[word] += 1
                    # 创建索引
                    if stem_word not in clean_words:
                        clean_words.add(stem_word)
                        if stem_word in index:
                            index[stem_word]["doc_id"].add(doc_id)
                        else:
                            index[stem_word] = {}
                            index[stem_word]["doc_id"] = set()
                            index[stem_word]["doc_id"].add(doc_id)
                tf_dict[doc_id] = term_fre
            len_dict[doc_id] = len(all_words)
            processed_doc[doc_id] = all_words
        # 存储文档平均长度
        avg_doclen = calculate_avg_doc_len(processed_doc)

        # 计算文档idf
        document_numbers = len(processed_doc)

        for term in index:
            df = len(index[term]["doc_id"])
            idf = math.log((document_numbers - df + 0.5) / (df + 0.5))
            index[term]["idf"] = idf

        save_index(index, "index.json", avg_doclen, tf_dict, len_dict)
    return index, processed_doc, avg_doclen, tf_dict, len_dict


# 清除标点符号以及数字
def clear_pun(document_content):
    clean_document = re.sub(r"[^\w\s]|[\d]", '', document_content)
    return clean_document


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


def save_index(index, file_path, avg_doclen, tf_dict, len_dict):
    for term, value in index.items():
        value["doc_id"] = list(value["doc_id"])
    data = {
        "avg_doclen": avg_doclen,
        "index": index,
        "tf_dict": tf_dict,
        "len_dict": len_dict
    }
    with open(file_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_index(file_path):
    print("Loading BM25 index from file, please wait.")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    avg_doclen = data['avg_doclen']
    index = data['index']
    tf_dict = data['tf_dict']
    len_dict = data['len_dict']

    return index, avg_doclen, tf_dict, len_dict


def read_documents_info(documents_folder):
    document_collection = {}  # 存储文档集合的哈希表
    # 大文件夹路径

    # 遍历大文件夹中的小文件夹
    for folder_name in os.listdir(documents_folder):
        folder_path = os.path.join(documents_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
        # 遍历小文件夹中的文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                # 将文字都转换为小写
                document_content = file.read().lower()
                # 将文件内容添加到文档集合中
                if len(document_content) != 0:
                    document_collection[file_name] = document_content
    return document_collection


# 读取stopword文件
def read_stopword_file(stopwords_file_path):
    with open(stopwords_file_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords


# BM25 model
def bm25_model(query, index, k, b, avg_doclen, tf_dict, len_dict):
    scores = {}
    for term in query:
        if term in index:
            idf = float(index[term]["idf"])
            for doc_id, doc_len in len_dict.items():
                if term in tf_dict[doc_id]:
                    tf = tf_dict[doc_id][term]
                else:
                    tf = 0
                score = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * (int(doc_len) / float(avg_doclen))))
                if doc_id in scores.keys():
                    scores[doc_id] += score
                else:
                    scores[doc_id] = score
    sorted_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_documents


def calculate_avg_doc_len(documents_clean):
    documents_numbers = len(documents_clean)
    total_word_numbers = 0
    for doc_id, doc_content in documents_clean.items():
        word_numbers = len(doc_content)
        total_word_numbers += word_numbers
    avg_doclen = total_word_numbers / documents_numbers
    return avg_doclen

# interactive()
automatic()
