# Lab 3: Facial recognition
## Advanced Machine Learning
### Josep de Cid Rodríguez
------
#### Included files:
- main.py ---> Executes the lab code. Parameters:
    - InceptionV3 graph path
    - Dataset path
    - **\-\-v** for verbose mode (Information)
    - **\-\-vv** for very verbose mode (Information and Debug)
    - **\-\-test test_set_path** by default to *"./test_set.csv"*
- model.py ---> Contains the Implementation in TF of our model, with a method to *train*, *create minibatches*...
- dataloader.py ---> Contains helper functions to load images, cache, and parse datasets.
- inception_cached.pkl ---> Cached output of InceptionV3 network for dataset images. I've decided to include this one as it takes a lot of time to pipe all the images and the first epoch takes around 30-40min. If you want to try the full process, feel free to delete this file.
------
#### Notes:
- Implemented Early stopping
- Implemented Annealing Learning Rate
- Using Verbose mode you can get some feedback about the cached images and the steps that the model is performing.
- Max accuracy obtained in 447 epochs: 85.12499928474426%