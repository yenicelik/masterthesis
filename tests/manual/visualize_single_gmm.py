import tensorflow as tf

from src.functional.pdf import pdf_gmm_diagional_covariance
from src.synthetic.generate_embeddings import generate_synthetic_embedding
from src.visualize.visualize_gmm import plot_contour3d_input2d

if __name__ == "__main__":
    print("Visualizing a single Gaussian Mixture Model")

    dimensions = 2
    src_components = 10
    tgt_components = 5

    src_emb_mus, src_emb_sigmas = generate_synthetic_embedding(
        d=dimensions,
        components=src_components
    )
    tgt_emb_mus, tgt_emb_sigmas = generate_synthetic_embedding(
        d=dimensions,
        components=tgt_components
    )

    #########################################
    # Visualize the first sampled Gaussian
    #########################################

    # We will ignore the covariance for the
    # calculate of the plotting boundaries!
    quadratic_scale = 20

    print("Source and sigma embeddings are: ")

    # mus = [tf.expand_dims(emb_src[0][i, :], axis=0) for i in range(src_components)]
    # mus_src = [src_emb_mu[i] for i in range(src_components)]
    # covs_src = [src_emb_sigma[i] * tf.eye(dimensions) for i in range(src_components)]

    # mus_tgt = [tgt_emb_mu[i] for i in range(src_components)]
    # covs_tgt = [tgt_emb_sigma[i] * tf.eye(dimensions) for i in range(src_components)]

    # print("Mus and cov are: ", mu.shape, cov.shape)

    plot_contour3d_input2d(
        pdf=lambda X: pdf_gmm_diagional_covariance(src_emb_mus, src_emb_sigmas).prob(X),
        x0_min=-1. * quadratic_scale,
        x0_max=1. * quadratic_scale,
        x1_min=-1. * quadratic_scale,
        x1_max=1. * quadratic_scale,
        ringcontour=False
    )