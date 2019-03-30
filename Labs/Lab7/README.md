# Autoencoders
## Josep de Cid | AML - Innopolis University

----
### Implementation

Once implemented the basic version adding some dense layers I've generalized the code allowing parametrized hyperparameters (for random search):

- **layers_enc:** #Layers in encoding part, with its corresponding #neurons and activation functions.
- **layers_dec:** #Layers in encoding part, with its corresponding #neurons and activation functions.
- **batch_size:** [32, 64, ..., 512]
- **latent_space_size:** latent_dim,
- **learning_rate:** Uniform LR in [0.0001, 0.1]

**NOTE:** Usually, Autoencoders have symmetric architecture but it's not mandatory and as long as we have a lot of possible combinations in the random search,
we've decided to allow variation in the encoder/decoder parts.