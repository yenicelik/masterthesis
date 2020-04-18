"""
    Learning a latent meaning space,
    by using metric learning,
    and learning a subspace which differentiates between different meanings

    I think a siamese network would be really good, because then we can locally
    and for multiple words train meaning-differences.

    pydml requires to have the full domain in the first place I believe,
    and we can also implicitly encode a dimensionality reduction technique here.

    Look at the principal component of the developed linear map to detect directions of most variance?

"""
from src.resources.corpus_semcor import CorpusSemCor
from src.resources.samplers import sample_semcor_data


def main():

    # Sample word vectors from semcor

    n = 10

    corpus = CorpusSemCor()
    # Sample as many samples as possible
    X, sentences_wordnet_cluster, sentences = sample_semcor_data(tgt_word='bank', n=n)

    print("Sentences and wordnet idx are")
    print(sentences)
    print(X.shape)
    print(sentences)

    # Now apply the metric learning algorithm to see if we can learn a model which satisfies this





    # Now sample amongst the image of the matrix




if __name__ == "__main__":
    print("Starting to learn the space")
    main()
