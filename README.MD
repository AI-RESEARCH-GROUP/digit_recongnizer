# INTRODUCTION

This is a handwritten numeral recognition project.The data contain gray-scale images of hand-drawn digits, from zero through nine.Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

# Python version 
``` 
3.6
```

# Download data
### data share from kaggle
``` 
    share link：https://www.kaggle.com/c/digit-recognizer
```
after downloading data, and save to "F:\PL\Digit_Recongnizer\data"

### csv field meaning
```
there are 3 files,contain test.csv;train.csv;sample_submission.csv.
    In the train.csv file, 
        there are 785 columns, including 784 pixel values and a tag value.
        And 42001 rows, which is 42001 digital photos for train.
    In the test.csv file, 
        there are 784 columns, including 784 pixel values. 
        And 28002 rows, which is 28002 digital photos for the test
    In the sample_summit.csv file, 
        there are two columns containing the image symbol and the predicted value
```

# Install Dependencies

### gpu mode
```
# install pytorch
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# then install other package
pip install -r requirements-cuda101.txt
```

### cpu mode
```
# install pytorch
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
# then install other package
pip install -r requirements.txt
```

# Preprocess
```
cd project_root_dir
python preprocess.py
```

# How to run 

### gpu mode
```
# defult mode is gpu mode

cd project_root_dir  
python train.py --gpu=0
python test.py --gpu=0
```

### cpu mode
```
cd project_root_dir 
python train.py --gpu=-1
python test.py --gpu=-1
```

* more args can be seen from src/args_and_config/args.py 
* more config can be seen from src/args_and_config/config.py 