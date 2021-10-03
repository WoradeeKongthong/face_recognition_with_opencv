from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import time
import cv2 
import numpy as np
import datetime

def get_person_images(search_name, save_name, num_images):
    # define driver
    driver = webdriver.Chrome("/home/samantha/chromedriver")
    driver.maximize_window()

    # go to the page
    driver.get("https://google.com")
    
    # find search box
    box = driver.find_element_by_xpath("/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input")

    # search and go to next page
    box.send_keys(search_name)
    box.send_keys(Keys.ENTER)
    
    # find element (img) and save the screenshot
    driver.find_element_by_xpath('//*[@id="hdtb-msb"]/div[1]/div/div[2]/a').click()
    time.sleep(3)
    
    # scroll page for n times
    pixel = 1000
    for i in range(5):
        driver.execute_script("window.scrollTo(0,{})".format(pixel)) 
        time.sleep(3)
        pixel = pixel + 10000
    
    driver.execute_script("window.scrollTo(0,0)")
    
    folder_name = f"data/faces/{save_name}"
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    for i in range(1, num_images+1):
        try :
            driver.find_element_by_xpath(f'//*[@id="islrg"]/div[1]/div[{i}]/a[1]/div[1]/img').screenshot(folder_name+f'/{save_name}_{i}.png')
            time.sleep(3)
        except :
            pass
    driver.close()

def get_person_face(imageFileName, face_cascade, scaleFactor, minNeighbor):
    # load image
    img = cv2.imread(imageFileName)
    # gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detection
    face_detect = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbor)
    # extract face from image
    if len(face_detect) == 1:
        x, y, w, h = face_detect[0]
        face = img[y:y+h, x:x+w]
        # save face instead of image
        cv2.imwrite(imageFileName, face)
    else:
        os.remove(imageFileName)

def create_face_dataset(search_names, save_names, img_num):
    for i in range(len(search_names)):
        get_person_images(search_names[i], save_names[i], img_num)
    
    # get image file name list
    fileNames = []
    for root, subdir, files in os.walk('data/faces'):
        for file in files:
            #if not file.startswith('haarcascade'):
            fileName = os.path.join(root, file)
            fileNames.append(fileName)
    
    # get xml for haarcascade
    face_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml')
    scaleFactor = 1.1 #default 1.1
    minNeighbor = 3 #default 3
    
    # extract faces
    for fileName in fileNames:
        get_person_face(fileName, face_cascade, scaleFactor, minNeighbor)

def create_array_dataset(name_list):
    # create dataset (X,y) fro training
    images = []
    labels = []
    name_int_map = {idx:name for idx,name in enumerate(name_list)}
    for i, name in enumerate(name_list):
        img_list = os.listdir('data/faces/'+name)
        path_list = [os.path.join('data/faces',name,i) for i in img_list]
        for imgPath in path_list:
            img = cv2.imread(imgPath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64,64))
            images.append(gray)
            labels.append(i)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, name_int_map

def face_recognition_on_video(videoPath, classifier, detector_scaleFactor, detector_minNeighbor, confThresh):
    # create face detector
    face_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml')

    # read video
    cap = cv2.VideoCapture(videoPath)
    while cap.isOpened():
        success, img = cap.read()
        if success :
            # gray image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detection
            face_detect = face_cascade.detectMultiScale(gray, detector_scaleFactor, detector_minNeighbor)
            for x,y,w,h in face_detect:
                # recognition
                id, conf = classifier.predict(gray[y:y+h, x:x+w])
                if conf > confThresh :
                    # draw box and label
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),1)
                    cv2.putText(img, f"{name_int_map[id]}:{round(conf,2)}", (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
            cv2.imshow("Result", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 break
        else :
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    # determine search names and save names
    search_names = ['jisooblackpink','jennieblackpink','roseblackpink','lisablackpink']
    save_names = ['Jisoo', 'Jennie','Rose','Lisa']
    
    # get person image and save
    #create_face_dataset(search_names, save_names, 250)
    
    # create training dataset
    X_train, y_train, name_int_map = create_array_dataset(save_names)
    
    # create the recognizer
    clf = cv2.face.LBPHFaceRecognizer_create()
    
    # train the recognizer
    clf.train(X_train, y_train)
    
    # save weights
    clf.write(f"data/saved_clf/blackpink_classifier_{str(datetime.datetime.now())}.xml")
    
    # load saved classifier
    #clf.read('saved_clf/blackpink_classifier_2021-10-03 20:07:07.040626.xml')
    
    # face recognition in video
    videoPath = 'data/video/bp.mp4'
    face_recognition_on_video(videoPath,clf, 1.5, 10, 100)

