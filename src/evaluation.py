def calculate_retrieved_and_relevant(retrieved, relevant):
    '''
    :retrieved: list of neighbours retrieved by DP algorithm
    :relevant: list of true nearest neighbours

    Calculates the number of results that are both retrieved and relevant.
    '''
    total = 0
    for x in retrieved:
        for y in relevant:
            total += x == y
    return total

def calculate_precision(retrieved, relevant):
    '''
    Calculates precision: percentage of retrieved that are relevant
    '''
    retrieved_and_relevant = calculate_retrieved_and_relevant(retrieved, relevant)
    return retrieved_and_relevant / len(retrieved)

def calculate_recall(retrieved, relevant):
    '''
    Calculates recall: percentage of relevant that are retrieved
    '''
    retrieved_and_relevant = calculate_retrieved_and_relevant(retrieved, relevant)
    return retrieved_and_relevant / len(relevant)

def calculate_top_k_accuracy(retrieved, relevant, k):
    '''
    Top k accuracy: 1 if any of the top k nearest neighbours are found, 0 o/w.
    '''
    return calculate_recall(retrieved, relevant[:k]) > 0

def calculate_mean(lst):
    '''
    Simple mean calculation
    '''
    return sum(lst) / len(lst)


def evaluate(eps_to_results, eps, retrieved, relevant):
    '''
    Evaluates everything and add to results dictionary
    '''
    raw_acc = calculate_top_k_accuracy(retrieved, relevant, 1)
    top_5_acc = calculate_top_k_accuracy(retrieved, relevant, 5)
    precision = calculate_precision(retrieved, relevant)
    recall = calculate_recall(retrieved, relevant)

    eps_to_results[eps]['raw_acc'].append(raw_acc)
    eps_to_results[eps]['top_5_acc'].append(top_5_acc)
    eps_to_results[eps]['precision'].append(precision)
    eps_to_results[eps]['recall'].append(recall)

    return eps_to_results
