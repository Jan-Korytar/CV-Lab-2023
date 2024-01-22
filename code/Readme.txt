Short instructions:
- if you'd like to invoke our model on a single focal stack, please use test.py
- to invoke on our test set, please use evaluate.py
- please find other details below

Environment settings:
- Create conda environment: conda env create -f environment.yml
- Activation: conda activate cvlab_pix2pix

-----------------------------------------------------------------------------------------
- these are the downloadabe assets that didn't fit on Moodle
- please extract into the "code" folder

Dataset:
- integrals_groundtruths.7z         includes training dataset and test ground truth images
    - download link (11.35 GB): https://mega.nz/file/fpg2TbiL#f-59mVjWQYvMUagsJCqrBrfB7tbBZwpBsZLfjxSW7mQ
    - integrals/                our integral images for training
    - groundtruths/             provided ground truth images

Test set:
- test_set.7z                       includes test set (5500 images from Part1 of batch_20230919) that our model can accept and have never seen
    - download link (3.51 GB): https://mega.nz/file/24Iw0BLD#XeEQVECpR_e1J5mWKcrTB_sjwAuKwjvwCIIfFYBVKRA
    - test/                         horizontally combined images as per pix2pix format
    - groundtruths/                 ground truth images for corresponding input image

Trained model weights:
- weights.zip
    - download link (202.6 MB): https://mega.nz/file/i5ZQnSpZ#nGKaKbboz4B0kRqC8CKohcW43GEyg-2xqTBxx3ixnv4
    - weights/                      includes our best model checkpoints
        - latest_250_G.pth          generator model checkpoint at epoch 250
        - latest_250_D.pth          discriminator model checkpoint at epoch 250

-------------------------------------------------------------------------------------------
Base folder contents:
- environment.yml                   conda environment required to run project
- integrator.py                     integrates simulated dataset to our focal length
- preprocessor.py                   preprocesses integrated and ground truth images for pix2pix accceptable format
- train.py                          runs the training and saves results in checkpoints folder
- test.py                           runs our pretrained model on three integrals with focal stack images [-0.1, -0.8, -1.6] and saves results in "test_results" folder. see generated folders info below
    - usage: python test.py [focal_0.1_image_path] [focal_0.8_image_path] [focal_1.6_image_path]
- evaluate.py                       runs our pretrained model on test set, generates test set restored images, and calculates average SSIM score on all of them
- pytorch-CycleGAN-and-pix2pix/     a clone of our fork https://github.com/kristofmaar/pytorch-CycleGAN-and-pix2pix/ please note that most of this were not authored by us, it's just required to run our code! see contributions on Github. 
- train_set_preprocessor.py         prepare train set for testing
- train_set_evaluate.py             we used this code to evaluate SSIM on train set

Generated folders:
- checkpoints/                      includes model that was trained by train.py
    - latest_net_G.pth              latest generator model checkpoint
    - latest_net_D.pth              latest discriminator model checkpoint
    - loss_log.txt                  log of losses during training
    - train_opt.txt                 export of training options
- test_results/weights/test_250     test results that are generated with test.py on pretrained model
    - images/
        - combined_image_fake.png   output of the model (restored image)
        - combined_image_real.png   input of the model
    - index.html                    openable to show results
- result/weights/test_250           test set restored images generated with evaluate.py on pretrained model
    - images/                       
        - [input image]_fake.png    output of the model (restored image)
        - [input image]_real.png    input of the model
    - index.html                    openable to show test results
- train_test/                       includes preprocessed images for train set testing