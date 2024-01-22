Environment settings:
- Create conda environment: conda env create -f environment.yml
- Activation: conda activate cvlab_pix2pix

Folder contents:
- integrator.py         integrates simulated dataset to our focal length
- preprocessor.py       preprocesses integrated and ground truth images for pix2pix accceptable format
- train.py              runs the training and saves results in checkpoints folder
- test.py               runs our pretrained model on three integrals with focal stack [-0.1, -0.8, -1.6]