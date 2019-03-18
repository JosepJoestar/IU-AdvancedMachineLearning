# Lab 4: Word2Vec
## Advanced Machine Learning
### Josep de Cid Rodríguez
------
#### Included files:
- main.py ---> Executes the lab code. Parameters:
    - **\-\-v** for verbose mode (Information)
    - **\-\-vv** for very verbose mode (Information and Debug)
- Word2Vec.py ---> Contains the Implementation in TF of our model, with a method to *train*, *create minibatches*...
- DataLoader.py ---> Contains helper functions to load data, generate triplets, parse dataset...
- Vocabulary.py ---> Data Structure to store maps from word to id, counters...
- Checkpoints ---> Checkpoints for Tensorboard visualisation
------
#### Notes:
- Checkpoints for fewer iterations, due to high computational requirements
- In Tensorboard Projector, the Variable that contains the correct visualisation is *Variable 5000x50*, not embedding, which is loaded first.
- I also attach some screenshots for the loss graph and some projector examples if there is some problem running it.
- Screenshots have been generated between 2nd-3rd epoch, to have some examples to show. Final results may vary.
- Checkpoints size are over 20MB (Moodle limit). There are some zips attached for each checkpoint. Those and TF_Info.zip should be extracted in the same folder.