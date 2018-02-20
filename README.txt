Dataset
  Dataset can be created by:
  (1) merge them together
     $ cat xaa xab xac > data.tar.gz

  (2) unzip the compressed file
     $ tar zxvf data.tar.gz
 
Trained Model
  The model is trained by the above data with 360 units, 800 epochs, and 30 samples in a batch.
  mdl360_800_30.h5
  mdl360_800_30.yaml
  
Whole Images
  One can try the model on the test files with prefix For_PRL_.
  For_PRL_a002_03.png
  For_PRL_a002_08.png
  For_PRL_NIST_14.jpg
  
Code for prediction
  One can use the trained model to do prediciton on previous images.
  Usage:
    python3 PRL_load_model_run.py mdl360_800_30 For_PRL_a002_03.png
