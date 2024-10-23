def read_ret(results_path):
    with open(results_path, 'r') as f:
        lines = f.readlines()
    ret_dict = {}
    for line in lines:
        results_spilt = line.split()
        query_id = results_spilt[0]
        rank = int(results_spilt[1])
        result_doc_id = results_spilt[2]
        if query_id in ret_dict:
            ret_dict[query_id][rank] = result_doc_id
        else:
            ret_dict[query_id] = {}
            ret_dict[query_id][rank] = result_doc_id
    return ret_dict


def read_rel(grels_path):
    with open(grels_path, 'r') as f:
        qrels = f.readlines()
    rel_dict = {}
    for relevance in qrels:
        rel_spilt = relevance.split()
        query_id = rel_spilt[0]
        rel_doc_id = rel_spilt[2]
        if query_id in rel_dict:
            rel_dict[query_id].add(rel_doc_id)
        else:
            rel_dict[query_id] = {rel_doc_id}
    return rel_dict


def calculate_precision(results, relevance):
    # Calculate Precision
    total_precision = 0.0
    total_queries = len(results)

    for query_id, query_results in results.items():
        results_set = set(query_results.values())
        if query_id in relevance:
            relevance_set = relevance[query_id]
            precision = len(results_set.intersection(relevance_set)) / len(results_set)
        else:
            precision = 0
        total_precision += precision

    precision = total_precision / total_queries
    return precision


def calculate_recall(results, relevance):
    # Calculate Precision
    total_recall = 0.0
    total_queries = len(results)

    for query_id, query_results in results.items():
        results_set = set(query_results.values())
        if query_id in relevance:
            relevance_set = relevance[query_id]
            recall = len(results_set.intersection(relevance_set)) / len(relevance_set)
        else:
            recall = 0
        total_recall += recall

    recall = total_recall / total_queries
    return recall


def calculate_p_10(results, relevance):
    total_p_10 = 0.0
    total_queries = len(results)

    for query_id, query_results in results.items():
        results_doc_id = set()
        if query_id in relevance:
            relevance_set = relevance[query_id]
            for i in range(1, 11):
                results_doc_id.add(query_results[i])
            p_10 = len(results_doc_id.intersection(relevance_set)) / 10
        else:
            p_10 = 0
        total_p_10 += p_10
    p_10 = total_p_10 / total_queries
    return p_10


def calculate_r_precision(results, relevance):
    total_r_precision = 0.0
    total_queries = len(results)

    for query_id, query_results in results.items():
        results_doc_id = set()
        if query_id in relevance:
            relevance_set = relevance[query_id]
            for i in range(1, len(relevance_set) + 1):
                results_doc_id.add(query_results[i])
            r_precision = len(results_doc_id.intersection(relevance_set)) / len(relevance_set)
        else:
            r_precision = 0
        total_r_precision += r_precision
    r_precision = total_r_precision / total_queries
    return r_precision


def calculate_map(results, relevance):
    total_queries = len(results)
    map_total = 0.0
    for query_id, query_results in results.items():
        average_precision = 0
        num_correct = 0
        for rank, doc_id in query_results.items():
            if doc_id in relevance[query_id]:
                num_correct += 1
                average_precision += num_correct / int(rank)
        if len(relevance[query_id]) > 0:
            average_precision /= len(relevance[query_id])
        map_total += average_precision
    average_precision = map_total / total_queries
    return average_precision


def calculate_brepf(results, relevance):
    total_queries = len(results)
    brepf_total = 0.0
    for query_id, query_results in results.items():
        brepf = 0.0
        results_set = set(query_results.values())
        relevance_set = relevance[query_id]
        relevant_number = len(results_set.intersection(relevance_set))

        non_relevant_number = 0
        for rank, doc_id in query_results.items():
            if doc_id in relevance[query_id]:
                brepf += 1 - non_relevant_number/relevant_number
            else:
                if non_relevant_number == relevant_number:
                    non_relevant_number = relevant_number
                else:
                    non_relevant_number += 1
        if relevant_number > 0:
            brepf /= relevant_number
        brepf_total += brepf
    brepf = brepf_total / total_queries
    return brepf



relevance = read_rel("files/qrels.txt")
results = read_ret("files/results.txt")
print(calculate_precision(results, relevance))
print(calculate_recall(results, relevance))
print(calculate_p_10(results, relevance))
print(calculate_r_precision(results, relevance))
print(calculate_map(results,relevance))
print(calculate_brepf(results,relevance))