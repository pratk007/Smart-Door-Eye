# import Librarys
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os

from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired


app=Flask(__name__)
camera = cv2.VideoCapture(0)

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'Faces_for_Recognition'

# Creating Class to upload folder
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

# Taking uploaded images one by one for training
path='Faces_for_Recognition'
images=[]
known_face_names=[]
myList=os.listdir(path)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    known_face_names.append(os.path.splitext(cl)[0])


# Function to find encodings of the uploaded images
def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
known_face_encodings=findEncodings(images)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Creating a function for face recognition and object and expression detection
def gen_frames():
    thres = 0.45 # Threshold to detect object
    camera.set(3,1280)
    camera.set(4,720)
    camera.set(10,70)
    
    classNames= []
    classFile = 'ObjectDetectionLibrarys/coco.names'
    with open(classFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    
    #confuguring path and using pre built librarys files for detection of various objects.
    configPath = 'ObjectDetectionLibrarys/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'ObjectDetectionLibrarys/frozen_inference_graph.pb'
    
    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)  


    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            classIds, confs, bbox = net.detect(frame,confThreshold=thres)

        # condition to check if there is atleast one face in the frame
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                cv2.rectangle(frame,box,color=(0,255,0),thickness=2)
                cv2.putText(frame,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)




            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left+12, top+12), (right+12, bottom+12), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.rectangle(frame, (right, top), (right+120, top-28), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                cv2.putText(frame, "Normal", (right + 4, top - 4), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET',"POST"])
def home():
    # uploading and saving the images taken 
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) 
    return render_template('index.html', form=form)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def results():
    return render_template('index_results.html')
    
if __name__=='__main__':
    app.run(debug=True)