Name: Goodreads Personal Recommendation Script
Author: Thalia Lightstone 

Description: 
There are two parts to this project
1. fetchgenre.py
    A script that takes in an unedited Goodreads exported CSV file and adds a column of the genres from OpenLibrary API using the ISBN. Then saves your data into a new CSV file.

2. collaborativerec.py
This script takes in two Goodreads exported CSV files, with an added column of genres from the above script. Then it provides recommendations for User 2 based on User 1's preferences using TensorFlow. 

Software requirements: 
I have a Mac with an M1 chip, so these were the installations I needed to run these scripts. 

pip install tensorflow-macos tensorflow-metal
pip install scikit-learn pandas

Additonally, 
You need to make sure your python version is 3.9 

python --version

If you have a different Mac or a PC, looking up how to run tensorflow on your computer would be necessary before running collaborativerec.py 


In this repository, I will include my goodreads data, but not my friends'. So, if you want to run this, you only need to provide one more dataset. I will provide the version that already has genre tags. 

