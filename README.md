# Image-Classifier-AIPND
Artifical Intelligence Programming with Python (AIPND) Project 2: Create Your Own Image Classifier Project Udacity

This Project is the second project from the AIPND Nannodegree Program from Udacity. Completed and submitted in accordance with the rubric from this project.

Project Summary:
  1. This project aims to predict the name of a flower based on this dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html.
  2. This project uses a pretrained model for image recognition.
  3. Accuracy of ~95% achieved using a pretrained Densenet201 model in 8 epochs.

Usage:
  1. Run the Jupyter Notebook for walkthrough of the training and end results; or:
  2. Run train.py on command line followed by predict.py, e.g.: 
     - *python train.py --arch "densenet121" --epochs 5*
     - *python predict.py --filepath 'flowers/1/image_06743.jpg*
