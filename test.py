def evaluate_results():
    # 读取结果文件
    results_file_path = "files/results.txt"
    results = {}
    with open(results_file_path, "r") as file:
        for line in file:
            query_id, _, doc_id, _ = line.strip().split("\t")
            if query_id in results:
                results[query_id].append(doc_id)
            else:
                results[query_id] = [doc_id]

    # 读取相关性判断文件
    relevance_file_path = "files/qrels.txt"
    relevance_judgments = {}
    with open(relevance_file_path, "r") as file:
        for line in file:
            query_id, _, doc_id, relevance = line.strip().split()
            relevance_judgments.setdefault(query_id, {})[doc_id] = int(relevance)
    print(relevance_judgments)

    # 计算评估指标
    num_queries = len(results)
    precision_total = 0
    recall_total = 0
    p_at_10_total = 0
    r_precision_total = 0
    average_precision_total = 0
    bpre_total = 0

    for query_id in results:
        retrieved_docs = results[query_id]
        relevant_docs = relevance_judgments.get(query_id, {})
        num_relevant_docs = len(relevant_docs)
        num_retrieved_docs = len(retrieved_docs)

        # 计算 Precision
        precision = len(set(retrieved_docs) & set(relevant_docs)) / num_retrieved_docs
        precision_total += precision

        # 计算 Recall
        recall = len(set(retrieved_docs) & set(relevant_docs)) / num_relevant_docs
        recall_total += recall

        # 计算 P@10
        p_at_10 = len(set(retrieved_docs[:10]) & set(relevant_docs)) / 10
        p_at_10_total += p_at_10

        # 计算 R-precision
        if num_relevant_docs > 0:
            r_precision = len(set(retrieved_docs[:num_relevant_docs]) & set(relevant_docs)) / num_relevant_docs
        else:
            r_precision = 0
        r_precision_total += r_precision

        # 计算 Average Precision
        average_precision = 0
        num_correct = 0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                num_correct += 1
                average_precision += num_correct / (i + 1)
        if num_relevant_docs > 0:
            average_precision /= num_relevant_docs
        print(average_precision)
        average_precision_total += average_precision


        # 计算 Bpref
        bpre = 0
        if num_relevant_docs > 0:
            non_relevant_docs = set(retrieved_docs) - set(relevant_docs)
            num_non_relevant_docs = len(non_relevant_docs)
            bpre = num_relevant_docs / (num_relevant_docs + num_non_relevant_docs)
        bpre_total += bpre

    # 计算平均值
    precision_avg = precision_total / num_queries
    recall_avg = recall_total / num_queries
    p_at_10_avg = p_at_10_total / num_queries
    r_precision_avg = r_precision_total / num_queries
    average_precision_avg = average_precision_total / num_queries
    bpre_avg = bpre_total / num_queries

    # 打印评估指标
    print(f"Precision: {precision_avg}")
    print(f"Recall: {recall_avg}")
    print(f"P@10: {p_at_10_avg}")
    print(f"R-precision: {r_precision_avg}")
    print(f"MAP: {average_precision_avg}")
    print(f"Bpref: {bpre_avg}")

# 调用函数
evaluate_results()
