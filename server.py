from flask import Flask, request, jsonify, send_file
from firebase_admin import credentials,initialize_app,storage
import cv2
import pickle
import mediapipe as mp
import pandas as pd
import numpy as np
from flask_cors import CORS
from audio_extract import extract_audio
import imutils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os

app=Flask(__name__)
CORS(app)
#initialise the storage and the application
cred = credentials.Certificate("key.json")
initialize_app(cred, {'storageBucket': 'finalyear-679f6.appspot.com'})
@app.route("/")
def home():
    return "Home"

@app.route("/download-video/<user_id>", methods=["GET"])
def download_video(user_id):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=f"{user_id}/")  # Specify the folder prefix
    blobLength = 0
    for blobt in blobs:
        blobLength+=1
    blob = bucket.blob(f"{user_id}/InterviewVideo{blobLength}.webm")  # Specify the folder path
    someLoader = bucket.blob("haarcascade_frontalface_default.xml")

    someLoader.download_to_filename("haarcascade_frontalface_default.xml")
    # Download video to a temporary file
    blob.download_to_filename("temp_video.webm")
    #start of model
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    resultFinal =[]

    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)


    cap = cv2.VideoCapture("temp_video.webm")
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            results = holistic.process(image)
            # Make Detections
            
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Concate rows
                row = pose_row+face_row
                
    #             # Append class name 
    #             row.insert(0, class_name)
                
    #             # Export to CSV
    #             with open('coords.csv', mode='a', newline='') as f:
    #                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                 csv_writer.writerow(row) 

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                resultFinal.append([body_language_class,np.max(body_language_prob)])
                print(body_language_class, body_language_prob)
                
                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                cv2.rectangle(image, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(body_language_class)*20, coords[1]-30), 
                            (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            except:
                pass
                                
            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Raw Webcam Feed',cv2.WND_PROP_VISIBLE)<1:
                break

    cap.release()
    cv2.destroyAllWindows()

    # tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
    # results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), [640,480]).astype(int))
    resultpd=pd.DataFrame(resultFinal,columns=["Emotion","Score"])
    print(resultpd.head())
    resultSend = {"Distracted":0,"Interested":0}
    resultSend = {"x":None,"y":None}
    sonyResult={}
    resultDistractedScore = resultpd[resultpd["Emotion"]=="Distracted"]["Score"].mean()
    resultDistracted=resultpd[resultpd["Emotion"]=="Distracted"].shape[0]/resultpd.shape[0]
    resultHappy=resultpd[resultpd["Emotion"]=="Happy"].shape[0]/resultpd.shape[0]
    resultSad=resultpd[resultpd["Emotion"]=="Sad"].shape[0]/resultpd.shape[0]
    resultTensed=resultpd[resultpd["Emotion"]=="Tensed"].shape[0]/resultpd.shape[0]
    sonyResult["Distracted"]={"x":"Distracted","y":resultDistracted}
    sonyResult["Happy"]={"x":"Happy","y":resultHappy}
    sonyResult["Sad"]={"x":"Sad","y":resultSad}
    sonyResult["Tensed"]={"x":"Tensed","y":resultTensed}
    # resultInterested =resultpd[resultpd["Emotion"]!="Distracted"] 
    # print(resultInterested==None)
    print(sonyResult)



    #----end of model sony---

    #start of model atp smile

    # load the face detector cascade and smile detector CNN
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = load_model("trained_model.h5")
    camera = cv2.VideoCapture("temp_video.webm")

    smile_count = 0
    total_frame_count = 0
        
    while True:
        # grab the next frame
        (grabbed, frame) = camera.read()

        if not grabbed:
            break
        
        total_frame_count += 1	

        # resize the frame to 300 pixels 
        frame = imutils.resize(frame, width=300) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert it to grayscale
        frameClone = frame.copy() # and then clone the original frame so we can draw on it later in the program

        # detect faces in the input frame, then clone the frame so that
        # we can draw on it
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # loop over the face bounding boxes
        for (fX, fY, fW, fH) in rects:
            # extract the ROI of the face from the grayscale image,
            # resize it to a fixed 28x28 pixels, and then prepare the
            # ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.astype("float") / 255.0 # Normalization to values btwn 0 n 1
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # determine the probabilities of both "smiling" and "not
            # smiling", then add to smile_count
            (notSmiling, smiling) = model.predict(roi)[0]
            if smiling > notSmiling:
                smile_count += 1 

    smile_ratio = (smile_count/total_frame_count)*100		
# if smile_ratio >= 20:
# 	print("Smiling well.", smile_ratio, smile_count,total_frame_count)
# else:
# 	print("Need to smile more.", smile_ratio, smile_count,total_frame_count)

# To Run: python smile_detection.py --cascade haarcascade_frontalface_default.xml --model model/trained_model.h5 --video videos/smiling_vid.mp4
    #---ENDDD of model atp smile----

    #----start of model sony speech transcription---
    # extract_audio(input_path="./temp_video.webm",output_path="./temp_audio.wav",output_format="wav")
    #---end of model sony speech transcription---
#START OF SDP MODEL FOR EYE GAZE DETECTION
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    min_eye_distance = 45  # Minimum distance between eyes for accurate detection
    min_neighbors_face = 5  # Minimum neighbors for face detection
    min_neighbors_eye = 5  # Minimum neighbors for eye detection
    threshold = 0.6 # Threshold for considering if facing the screen
    screen_ratio = 0.5 # Ratio of face width to consider as facing the screen

    # Start video capture from webcam
    capture = cv2.VideoCapture("temp_video.webm")

    total_frames = 0
    facing_screen_frames = 0
    not_facing_screen_frames = 0

    while True:
        # Read a frame from the video capture
        ret, frame = capture.read()

        if not ret:
            break

        # Increment total frames count
        total_frames += 1

        # Convert frame to grayscale for better performance in face and eye detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors_face)

        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Define region of interest (ROI) for eye detection within the face region
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=min_neighbors_eye)

            # Check if both eyes are detected within the face region
            if len(eyes) == 2:
                # Calculate the distance between the eyes
                eye_distances = [abs(ex1 - ex2) for (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) in zip(eyes, eyes[1:])]
                min_eye_distance_detected = min(eye_distances)

                # Check if the minimum eye distance meets the threshold
                if min_eye_distance_detected >= min_eye_distance:
                    # Calculate the ratio of face width
                    face_width_ratio = (eyes[-1][0] + eyes[-1][2] - eyes[0][0]) / w

                    # Check if the face width ratio meets the threshold for facing the screen
                    if face_width_ratio >= screen_ratio:
                        facing_screen_frames += 1
                    else:
                        not_facing_screen_frames += 1

        # Display the frame with face and eye detection
        # cv2.imshow('Gaze Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

    # Calculate the percentage of frames where the candidate is facing the screen
    percentage_facing_screen = round((facing_screen_frames / total_frames) * 100, 2)
    # print("Percentage of Time Facing the Screen:", percentage_facing_screen)
#END OF EYE GAZE DETECTION
    # start of sending result
    resultSendable={
        "DistractionScore":resultDistractedScore*100,
        "SmileScore":smile_ratio,
        "EyeScore":percentage_facing_screen

    }
    os.remove("haarcascade_frontalface_default.xml")
    #end of sending result
    return jsonify(resultSendable)

@app.route("/list-videos", methods=["GET"])
def list_videos():
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="videos/")  # Specify the folder prefix
    blobLength = 0

    video_list = []
    for blob in blobs:
        blobLength+=1
        video_info = {
            "name": blob.name,
            "bucket": blob.bucket.name,
            "size": blob.size,
            "length":blobLength,
            # Add other relevant metadata if needed (e.g., creation_time)
        }
        video_list.append(video_info)

    return jsonify(video_list)

@app.route("/extracter-audio",methods=["GET"])
def extracter_audio():
    import subprocess
    import requests

    video_path="./temp_video.webm"
    audio_path="./temp_audio.wav"
    ffmpeg_command = ["ffmpeg", "-i", video_path, "-vn", audio_path]
    subprocess.run(ffmpeg_command, check=True)
    #------Extraction done------
    #START TRANSCRIPTION
    upload_endpoint = "https://api.assemblyai.com/v2/upload"
    filename = "temp_audio.wav"
    def read_file(filename,chunk_size=5242880):
        with open(filename,'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    headers =  {'authorization':"b1c67689c9254de495e591ecf77132e8"}
    upload_response = requests.post(upload_endpoint,
                                    headers = headers,
                                    data=read_file(filename))
    print(upload_response.json())
    audio_url = upload_response.json()['upload_url']
    #transcription
    transcription_endpoint  = "https://api.assemblyai.com/v2/transcript"
    json={"audio_url":audio_url}
    transcript_response = requests.post(transcription_endpoint,json=json,headers=headers)
    print(transcript_response.json())
    transcript_id = transcript_response.json()['id']
    print(transcript_id)

    #polling
    def get_transcription_result_url(transcript_id):
        polling_endpoint = transcription_endpoint + "/"+ transcript_id

        while True:
            polling_response = requests.get(polling_endpoint,headers=headers)
            if polling_response.json()['status'] == 'completed':
                return polling_response.json(), None
            elif polling_response.json()['status'] == 'error':
                return polling_response.json(),polling_response.json()['error']
    transcribed_response,error = get_transcription_result_url(transcript_id)
    print(transcribed_response)

    if transcribed_response:
        text_filename =  filename + ".txt"

        with open(text_filename,'w') as f:
            f.write(transcribed_response['text'])
        print("Transcription Saved!!")
    elif error:
        print("Error!")

    #--------START OF FILLER WORD ANALYSIS------#
    data=None
    with open("temp_audio.wav.txt",'r') as f:
        data= f.read()
        print(data)
    from nltk.tokenize import regexp_tokenize

    filler_words = [
        "um", "uh", "like", "you know", "well", "so", "actually", "basically",
        "literally", "honestly", "i mean", "kind of", "sort of", "anyway", "right",
        "okay", "just", "totally", "absolutely", "really"
    ]

    # Create a regular expression pattern to tokenize text while preserving multi-word phrases
    pattern = "|".join(filler_words)  # Concatenate filler words with "|" to match any of them
    pattern = fr"\b{pattern}\b"  # Add word boundaries to match whole words only. r to avoid Python reading \b as back space

    def count_filler_words(text):
        # Tokenize the text using regexp_tokenize and the defined pattern
        tokens = regexp_tokenize(text.lower(), pattern)

        print(tokens)
        # Count occurrences of filler words
        filler_word_counts = {word: tokens.count(word) for word in filler_words}

        return filler_word_counts

    example_text = "Your example text goes here. You know, like like like, um, well, basically, uh, I mean, right?"
    filler_word_counts = count_filler_words(data)

    # Print results
    resultAudio=0
    for word, count in filler_word_counts.items():
        if count >= 3:
            resultAudio+=1
    #start of response relevance
    def calculate_similarity(question, response):
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Tokenize and encode the question and response
        question_tokens = tokenizer.encode(question, add_special_tokens=True)
        response_tokens = tokenizer.encode(response, add_special_tokens=True)

        # Convert token IDs to tensors
        question_tensor = torch.tensor([question_tokens])
        response_tensor = torch.tensor([response_tokens])

        # Forward pass through the BERT model
        with torch.no_grad():
            question_embedding = model(question_tensor)[0][:, 0, :].numpy()  # Take the [CLS] token embedding
            response_embedding = model(response_tensor)[0][:, 0, :].numpy()
        
        print("question_tokens: ",question_tokens, "\nquestion_tensor: " , question_tensor, "\nquestion_embedding: ", question_embedding)

        # Calculate cosine similarity
        similarity_score = cosine_similarity(question_embedding, response_embedding)[0][0]

        return similarity_score

    # Example usage
    question = "Are you a self-motivator?"
    response = "Absolutely. For me, internal motivation works far more than external motivation ever could."

    similarity_score = calculate_similarity(question, data)
    print("Similarity Score:", similarity_score)
    resultSendAudio={
        "FillerScore":resultAudio,
        "ResponseScore":similarity_score*100
    }
    os.remove("temp_audio.wav")
    os.remove("temp_audio.wav.txt")
    os.remove("temp_video.webm")
    return jsonify(resultSendAudio)



if __name__=="__main__":
    app.run(debug=True)