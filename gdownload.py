import gdownload

url = 'https://drive.google.com/file/d/1ytBWLPBDzOXHxm6jltFZHowVqjO3Vfqr/view?usp=sharing'
output = 'model/siamese_cosine_model_best3.h5'
gdownload.download(url, output, quiet=False)
