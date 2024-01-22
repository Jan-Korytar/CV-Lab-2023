import subprocess

command = [
    'python', 'pytorch-CycleGAN-and-pix2pix/train.py',
    '--dataroot', 'dataset_pix2pix/combined',
    '--name', 'thermal_focalstack_pix2pix',
    '--model', 'pix2pix',
    '--direction', 'BtoA',
    '--input_nc', '9',
    '--gpu_ids', '0',
    '--display_ncols', '-1',
    '--batch_size', '8',
    '--lr', '0.00010',
    '--n_epochs', '200'
]

subprocess.run(command)