import tensorflow as tf

from src.functional.pdf import pdf_gmm_diagional_covariance
from src.synthetic.generate_embeddings import generate_synthetic_embedding
from src.visualize.visualize_gmm import plot_contour3d_input2d

if __name__ == "__main__":
    print("Visualizing a single Gaussian Mixture Model")

    dimensions = 2
    src_components = 10
    tgt_components = 5

    emb_src = generate_synthetic_embedding(
        d=dimensions,
        components=src_components
    )
    emb_tgt = generate_synthetic_embedding(
        d=dimensions,
        components=tgt_components
    )

    #########################################
    # Visualize the first sampled Gaussian
    #########################################

    # We will ignore the covariance for the
    # calculate of the plotting boundaries!
    quadratic_scale = 20

    # mus = [tf.expand_dims(emb_src[0][i, :], axis=0) for i in range(src_components)]
    mus_src = [emb_src[0][i, :] for i in range(src_components)]
    covs_src = [emb_src[1][i, :] * tf.eye(dimensions) for i in range(src_components)]

    mus_tgt = [emb_tgt[0][i, :] for i in range(src_components)]
    covs_tgt = [emb_tgt[1][i, :] * tf.eye(dimensions) for i in range(src_components)]

    # print("Mus and cov are: ", mu.shape, cov.shape)

    plot_contour3d_input2d(
        pdf=lambda X: pdf_gmm_diagional_covariance(X, mus_src, covs_src),
        x0_min=-1. * quadratic_scale,
        x0_max=1. * quadratic_scale,
        x1_min=-1. * quadratic_scale,
        x1_max=1. * quadratic_scale,
        ringcontour=False
    )