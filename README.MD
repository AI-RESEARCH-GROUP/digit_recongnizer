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


# How to run 
```
python preprocess.py
python train.py
python test.py
```
