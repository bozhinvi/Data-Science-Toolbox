# Data Science Toolbox
This is a repository based on a summer school course I took in 2017 at the Barcelona School of Economics. I updated the code (but stikll working on it) and created a container which can be build using docker built command. The container can be used to play around with the code. Have in mind that all changes are not persistent on your local hard drive because the directory is not mounted but copied during docker build.

## Topics covered

Part 1 covers a simple Python introiduction as well as typical examples of classical supervised and unsupervised ML algorithms applied to classic datasets such as the iris dataset and the titanic dataset. The main purpose of this repo was to see how scikit-learn works and take a closer look on few simple algorithmis such as regression, support vecotr machines and desicions trees as well as KNN. We also covered clustering and text classification asthe last part of the course.

Part 2 was more exciting since we proceeded with Keras and TensowFlow and trained neural networks. I will publish it after I cleaned up the code which is from 2017 and has to undergo some serious updates due to depreciated packages.

## Using Docker to play around

If you like to play around with code without having to install any packages, you can use Docker to do so. Just start Docker and built the image and start it using the following code:

`docker build -t data_science_toolbox:0.1 .`

`docker run -d --name ds-toolbox -p 8888:8888 data_science_toolbox:0.1`

