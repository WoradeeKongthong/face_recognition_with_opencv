# face_recognition_with_opencv  
In the python scrips includes  
## scraping images of person via person's name
- First specify a list of search keyword for person images. 
- Specify a list of name as a label.
## create dataset of cropped faces
- You need number of scraped images for each person. You should put the number higher than the number of actual images you want because each time of scraping might get some error.
- Then create the X and y for training the recognizer.
## train the opencv face recognition
- The recognizer I used is cv2.face.LBPHFaceRecognizer.
- Save the trained recognizer as an xml file
## do face recognition
- I perform the face recognition of my trained mdoel on a video.
