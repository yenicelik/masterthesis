"""
    Exhaustively looks through all possible models, and uses the CV-loss to determine if a clustering is available.
    We aggregate a dataset by taking both from the SemCor dataset, and enriching these with BERT embeddings.
    The validation loss then consists of how well the items in same clusters are actually put into the same buckets
"""


if __name__ == "__main__":
    print("Starting hyper-parameter search of the model")

