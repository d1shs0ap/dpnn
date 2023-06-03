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
