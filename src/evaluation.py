def extract_location_set(kd_node):
    '''
    Given a list of KD nodes, extract their locations and put them into a set
    '''
    return set(tuple(node.data) for node in kd_node)

def calculate_retrieved_and_relevant(retrieved, relevant):
    '''
    :retrieved: list of neighbours retrieved by DP algorithm
    :relevant: list of true nearest neighbours

    Calculates the number of results that are both retrieved and relevant.
    '''
    intersection = extract_location_set(retrieved) & extract_location_set(relevant)
    return len(intersection)

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

