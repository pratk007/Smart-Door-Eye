
# Smart Gate

Face detection is a widely growing technology and now it is in high demand for security purposes. This project  is a prototype that demonstrate real world secenario which put a first line of defence system to main door of urban houses. The prototype is deployed outside the main door of house trained with photo's of all the family members that allows a person inside the home only if he/she is a family-member otherwise it gathers all the details of person and displays it and not only that it also warns if something unforseen situatuion is there and also keeps a track of it into the database. The details that it detect of a stranger are facial expressions, objects carried by them, time, photos and based on these by supervised learning it also predicts the person i.e. criminal, delivery boy, milk-man or newspaper vendor.

## Tech Stack

**Frontend**:  HTML, CSS, JavaScript

**Backend:**  Flask, JavaScript

**Database:** MySQL

**Database Connection:**  SQLAlchemy

**ML Models:**  Python(Open-CV, Tensorflow)


## Installation

At first Clone my repository

Install Dependencies using pip 

```bash
  pip install reqirenments.txt --user
```
    
## Deployment

To deploy this project run the file

```bash
  main.py
```
Then go to the url:
```bash
  http://127.0.0.1:5000
```
Then see the results on:
```bash
  http://127.0.0.1:5000/results
```

## Features

- Detects the face of family member(Trained already)
- Note downs the details of stranger
- Details that it tracks : Facial Expression,objects carried,Timings, photo
- Based on these details predicts the person.


## Usage/Examples

In Urban areas, it can serve as the most reliable security technology for people and it will reduce violence up to a very far extend.


## Demo

Insert gif or link to demo


## Acknowledgements

 - [Face Recognition](https://github.com/ageitgey/face_recognition)
 - [opencv for computer vision](https://github.com/opencv/opencv)
 - [Tensorflow for object detection](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/)

