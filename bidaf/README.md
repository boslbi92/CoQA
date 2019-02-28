# bidaf-keras
Implementation of [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603) in Keras 2

## Release-info
This project is available for use as a complete module. Just install the requirements (will update with only the necessary ones) and then run the module named 'bidaf' using the command:
`python -m bidaf`

You can append your code in the \__main__.py file according to your needs. We are working on updating this as a perfect module which you can use via commandline arguments.

A pretrained model will be made available soon. My teammate is working on building this model in tensorflow and it will also be available soon.

## What you can do with this project:
- Train/Retrain this model with your own dataset.
- Use the pretrained model for extending your own model or to try this one out.
- Modify the code of this model to develop a new model architucture out of it.

## To-do list
- Make this project as a portable module and publish it on pypi
- Support to use module using web based GUI interface

I don't know which things should I keep in mind to do this (such as if a directory doesn't exists at runtime, the model should create it dynamically instead of throwing an error). If you have such points that I should keep in mind, consider contributing to this project. Also, if you have time to implement it in this project, submit your work with a pull request on a new branch.

Thoughts, samples codes, modifications and any other type of contributions are appreciated. This is my first project in Deep Learning and also first being open source. I will need as much help as possible as I don't know the path I need to follow. Thank you..

## Issues
- Open:
  1. https://github.com/keras-team/keras/issues/12263
- Solved:
  1. https://github.com/tensorflow/tensorflow/issues/24519
  2. https://github.com/keras-team/keras/issues/11978

## My team
- [Dharanee Patel](https://github.com/dharaneepatel15/)
- [Zeel Patel](https://github.com/zeelp898/)

## Our Guide
- [Prof. Tushar A. Champaneria](https://github.com/tacldce/)

## Special Thanks to the researchers of BiDAF
- Sir, [Minjoon Seo](https://github.com/seominjoon/) for helping me out personally
- The team of allenai (http://www.allenai.org/)