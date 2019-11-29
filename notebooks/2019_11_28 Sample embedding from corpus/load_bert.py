"""
    Includes the logic which samples a few embeddings for each word.
    This is similar to Monte-Carlo sampling to represent words.

    - Do 1000 times:
        - Find all occurences of "word" in BERT
        - Run BERT on the word (or ELMo if tokenization is easier)
    - Create a Probability distribution from the above samples
        (do we need a priior for this distribution...? or can we just generate a pointcloud with this)

    This assumes that the biLM outputs a single vector somehow which can then used to apply nearest-neighbor search
    But this is perhaps the idea behind this. Get individual word-embeddings through the initial one.
    -> Can then generate a thesaurus, or sample sentences where these words can be used within.

    Perhaps ELMo is a better choice. BERT seems a bit too "masked". However, ELMo does not have embeddings for all languages...
"""

