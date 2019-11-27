# Lines of work

## Word Emebddings

- Word2Vec x
- GloVe x
- Gaussian Embeddings x
  - symmetric similarity
  - asymmetric similarity metric
- fastText (sum of n-grams)
- Bayesian Skip-Gram model (Embedding Words as Distributions with a Bayesian Skip-gram Model)
- FRAGE, frequency agnostic word embeddings, as the word embeddings are strongly biased towards frequency (FRAGE: Frequency-Agnostic Word Representation)

- revealing similarity to eigenvectors (Interpreting Word Embeddings with Eigenvector Analysis)

## Gaussian Embeddings

- Word embeddings
- Collaborative filtering 
- Gaussian Embeddings with wasserstein distance loss (for embedding training) (Gaussian Word Embedding with a Wasserstein Distance Loss)
- Some first work in theoretical foundations using a convex relaxation of the problem at hand, and checking non-convex convergence rate (Provable Gaussian Embedding with One Observation) 

## Normalising Flows

- Planar Flow x
- Real-NVP x
- NICE 
- Glow x
- Flow++ x
- Inverse AR Flow 

## Bilingual Token Matching

All work using pre-trained embeddings

### Using discrete Word-Embeddings

- Find orthogonal matrix minimizing \forall x, y : xWy x
- Find orthogonal matrix through min-max game, where generator finds W, discriiminator says which original embedding it was from x. Apparently not robust when distant languages (Withouth parallel data)
- Project to common correlation subspace Z and then map (Generalizing Procrustes Analysis for Better Bilingual Dictionary Induction)
- Cyclic Loss through back-translation and projection to same subspace (Unsupervised Machine Translation Using Monolingual Corpora Only)
- Use sentence-translation models, and then infer bilingual tokens 
- Use different retrieval metric (instead of cosine distance) when training (Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion)
- Wasserstein and Procrustes with convex relaxation of the problem (Unsupervised Alignment of Embeddings with Wasserstein Procrustes)

### Using "smooth" / probabilistic word-embeddings

- Joint learning of word and topic embeddings using Wasserstein metric
- Gromov-Wasserstein Alignment of word-embedding spaces X - WY x
- Normalising flows between Gaussian embeddings, using W as a flow x

## Other (possibly related) work

- Gaussian Embedding of graphs, mostly in collaborative filtering (DEEP GAUSSIAN EMBEDDING OF GRAPHS: UNSUPERVISED INDUCTIVE LEARNING VIA RANKING, Graph Normalizing Flows)
- Muli-Modal-Domain-Translatoin (M 3 D-GAN: Multi-Modal Multi-Domain Translation with Universal Attention)
- All work with unpaired image to image translation
- Using normalising flows for latent-variable modeling in machine translation (Investigating the Impact of Normalizing Flows on Latent Variable Machine Translation)
- Normalising flow for unpaired image2image translation (ALIGNFLOW: LEARNING FROM MULTIPLE DOMAINS VIA NORMALIZING FLOWS)
- Normalising flows for clustering and classification, by projecting onto one p isotropic gaussians (Clustering and classification through normalizing flows in feature space)
- Task specific word embeddings, similar to transfer learning and fine-tuning (Simple task-specific bilingual word embeddings)
- Gaussian embeddings, somehow with CNNs, for collaborative filtering (Convolutional Gaussian Embeddings for Personalized Recommendation with Uncertainty)
