from src.functional.pdf import pdf_gaussian
from src.synthetic.generate_embeddings import generate_synthetic_embedding
from src.visualize.visualize_gmm import plot_contour3d_input2d

if __name__ == "__main__":
    print("Plotting Gaussian Mixutre Models GMMs using matplotlib")

    import tensorflow as tf

    dimensions = 2

    src_emb_mus, src_emb_sigmas = generate_synthetic_embedding(
        d=dimensions,
        components=10
    )
    tgt_emb_mus, tgt_emb_sigmas = generate_synthetic_embedding(
        d=dimensions,
        components=10
    )

    #########################################
    # Visualize the first sampled Gaussian
    #########################################

    mu = src_emb_mus[0]
    cov = src_emb_sigmas[0]

    print("Mus and cov are: ", mu.shape, cov.shape)

    # We will ignore the covariance for the
    # calculate of the plotting boundaries!
    quadratic_scale = 20

    plot_contour3d_input2d(
        pdf=lambda X: pdf_gaussian(X, mu, cov),
        x0_min=-1. * quadratic_scale,
        x0_max=1. * quadratic_scale,
        x1_min=-1. * quadratic_scale,
        x1_max=1. * quadratic_scale
    )
