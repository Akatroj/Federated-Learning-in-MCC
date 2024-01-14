dataset source: https://www.kaggle.com/datasets/naderabdalghani/iam-handwritten-forms-dataset
probably same as: https://paperswithcode.com/dataset/iam

Download it and extract in this directory. Then run `preprocess.py` which resizes the images, converts them to jpg, and picks a fraction of images from each author to end up with required number of images (see src). Images are renamed to "img_x_y.jpg", where x denotes image number and y denotes to which device it should be assigned in federated training.
