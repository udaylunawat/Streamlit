"""This is an Object detection and Optical Character Recognition(OCR) app that enables a user to

- Select or upload an image.
- Get a annotated and cropped license plate image.
- Play around with Image enhance options (OpenCV).
- Get OCR Prediction using various options.

"""

# streamlit configurations and options
import streamlit as st
from streamlit import caching
st.beta_set_page_config(page_title="Ex-stream-ly Cool App", page_icon="😎", layout="centered", initial_sidebar_state="expanded")
st.set_option('deprecation.showfileUploaderEncoding', False)





from PIL import Image,ImageEnhance
import re
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import time
import random
import pandas as pd

try:
    import easyocr
except:
    pass

# miscellaneous modules
# from pyngrok import ngrok
import webbrowser

# https://github.com/keras-team/keras/issues/13353#issuecomment-545459472
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "output/models/inference/plate_inference_tf2_2.h5": {
        "url": "https://storage.googleapis.com/dracarys3_bucket/Public/plate_inference_tf2_2.h5",
        "size": 146111272
    },
    "output/models/inference/yolov3-custom_last.weights": {
        "url": "https://storage.googleapis.com/dracarys3_bucket/Public/yolov3-custom_last.weights",
        "size": 246305388
    },
    "cfg/yolov3-custom.cfg":{
        "url": "https://raw.githubusercontent.com/udaylunawat/Automatic-License-Plate-Recognition/master/cfg/yolov3-custom.cfg"
    },
    "cfg/obj.names":{
        "url": "https://raw.githubusercontent.com/udaylunawat/Automatic-License-Plate-Recognition/master/cfg/obj.names"
    }
}

import urllib
# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    os.makedirs('/'.join(file_path.split('/')[:-1]))

    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        st.write("file name:{}, size:{}".format(file_path, os.path.getsize(file_path)))
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# Download external dependencies.info = st.empty()
if st.checkbox("MAYBE DOWNLOAD 250MB OF DATA TO THE SERVER. THIS MIGHT TAKE A FEW MINUTES!"):


    for filename in EXTERNAL_DEPENDENCIES.keys():
        print("file name:{}, size:{}".format(filename, os.path.getsize(filename)))
        download_file(filename)

#==================================== enhance.py =================================================

crop, image = None, None
img_size, crop_size = 600, 400


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image, rate):
    return cv2.threshold(image, rate, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#skew correction
def deskew(image):
  coords = np.column_stack(np.where(image > 0))
  angle = cv2.minAreaRect(coords)[-1]
  if angle < -45:
    angle = -(90 + angle)
  else:
      angle = -angle
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
  return rotated

def cannize_image(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def cropped_image(image, b):
    crop = cv2.rectangle(np.array(image), (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
    crop = crop[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
    crop = Image.fromarray(crop)

    return crop
    
# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def enhance_crop(crop):

    st.write("## Enhanced License Plate")
    rgb = np.array(crop.convert('RGB'))
    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    enhance_type = st.radio("Enhance Type",\
                                   ["Original","Custom", "Gray-Scale","Contrast",\
                                    "Brightness","Blurring","Cannize",\
                                    "Remove_noise", "Thresholding", "Dilate",\
                                    "Opening","Erode", "Deskew"])
    crop_display = st.empty()
    slider = st.empty()
    if enhance_type in ["Contrast","Brightness","Blurring","Thresholding","Custom"]:
        rate = slider.slider(enhance_type,0.2,8.0,(1.5))

    if enhance_type == 'Original':
        output_image = crop

    elif enhance_type == 'Gray-Scale':
        output_image = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

    elif enhance_type == 'Contrast':
        enhancer = ImageEnhance.Contrast(crop)
        output_image = enhancer.enhance(rate)

    elif enhance_type == 'Brightness':
        enhancer = ImageEnhance.Brightness(crop)
        output_image = enhancer.enhance(rate)

    elif enhance_type == 'Blurring':
        img = cv2.cvtColor(rgb,1)
        output_image = cv2.GaussianBlur(img,(11,11),rate)
    
    elif enhance_type == 'Cannize':
        output_image = cannize_image(crop)

    elif enhance_type == "Remove_noise":
        output_image = remove_noise(rgb)

    elif enhance_type == "Thresholding":
        output_image = thresholding(gray, rate)

    elif enhance_type == "Dilate":
        output_image = dilate(rgb)

    elif enhance_type == "Opening":
        output_image = opening(rgb)

    elif enhance_type == "Erode":
        output_image = erode(rgb)

    elif enhance_type == "Deskew":
        output_image = deskew(np.array(gray))

    elif enhance_type == "Custom":
        # resized = cv2.resize(gray, interpolation=cv2.INTER_CUBIC)
        dn_gray = cv2.fastNlMeansDenoising(gray, templateWindowSize=7, h=rate)
        gray_bin = thresholding(dn_gray, rate)
        output_image = gray_bin

    crop_display.image(output_image, width = crop_size, caption = enhance_type)
    return output_image


#============================== ocr.py =====================================
import pytesseract
try:
    import easyocr
    reader = easyocr.Reader(['en'])
except:
    pass

def try_all_OCR(crop_image):
    st.write(" **OEM**: Engine mode \
             \n**PSM**: Page Segmentation Mode \
             \n Click [here](https://nanonets.com/blog/ocr-with-tesseract/) to know more!")

    progress_bar = st.progress(0)
    counter = 0
    for oem in range(0,4):
        for psm in range(0,14):
            counter += 1
            try:
                custom_config = r'-l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem {} --psm {}'.format(oem,psm)
                text_output = pytesseract.image_to_string(crop_image, config=custom_config)
                st.warning("oem: {} psm: {}: {}".format(oem, psm, text_output))
                progress_bar.progress(counter/(4*14))
            except:
                continue

def easy_OCR(crop_image):
    ocr_output = reader.readtext(np.array(crop_image))
    text_output = ''
    for text in ocr_output:
        text_output += text[1]
    return text_output

def OCR(crop_image):
    text_output = ''
    try:
        custom_config = r'-l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem 1 --psm 6'
        text_output = pytesseract.image_to_string(crop_image, config=custom_config)
        print(custom_config,':',text_output)
    except:
        pass
    return text_output

#================================== retinanet_helper.py=================================

# Machine Learning frameworks
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

# https://github.com/keras-team/keras/issues/13353#issuecomment-545459472
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

from config import labels_to_names
from utils.enhance import cropped_image

crop, image = None, None

#================================= Functions =================================

@st.cache(suppress_st_warning=True, allow_output_mutation=False, show_spinner=False)
def load_retinanet():
    # caching.clear_cache()
    model_path = 'output/models/inference/plate_inference_tf2.h5'

    # load retinanet model
    print("Loading Model: {}".format(model_path))
    with st.spinner("Loading retinanet weights!"):
        model = models.load_model(model_path, backbone_name='resnet50')

    #Check that it's been converted to an inference model
    try:
        model = models.convert_model(model)
    except:
        print("Model is likely already an inference model")

    return model


def image_preprocessing(image):
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # Image formatting specific to Retinanet
    image = preprocess_image(image)
    image, scale = resize_image(image)
    return image, draw, scale

def load_image(image_path):
    image = np.asarray(Image.open(image_path).convert('RGB'))
    image = image[:, :, ::-1].copy()
    return image

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def inference(model, image, scale): # session
    # Run the inference
    
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    st.sidebar.success("Processing time for RetinaNet: {} {:.4f} seconds.".format("\n",time.time() - start))
    print("Processing time for RetinaNet: {:.4f} seconds ".format(time.time() - start))

    # correct for image scale
    boxes /= scale
    return boxes, scores, labels


def draw_detections(draw, boxes, scores, labels, confidence_cutoff):
    draw2 = draw.copy()
    crop_list = []
    b = None
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < confidence_cutoff:
            break

        #Add boxes and captions
        color = (255, 255, 255)
        thickness = 2
        b = np.array(box).astype(int)

        try:
            crop_list.append([cropped_image(draw2, (b[0], b[1], b[2], b[3]) ), score])

            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

            if(label > len(labels_to_names)):
                st.write("WARNING: Got unknown label, using 'detection' instead")
                caption = "Detection {:.3f}".format(score)
            else:
                caption = "{} {:.3f}".format(labels_to_names[label], score)

            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            # startX, startY, endX, endY = b[0], b[1], b[2], b[3]
        except TypeError as e:
            st.error("No plate detected")
        
    return (b[0], b[1], b[2], b[3]), draw, crop_list


def retinanet_detector(image_path, model, confidence_cutoff):

    image = load_image(image_path)
    
    image, draw, scale = image_preprocessing(image)
    boxes, scores, labels = inference(model, image, scale) # session
    b, draw, crop_list = draw_detections(draw, boxes, scores, labels, confidence_cutoff)

    #Write out image
    drawn = Image.fromarray(draw)

    # draw.save(image_output_path)
    # print("Model saved at", image_output_path)

    return drawn, max(scores[0]), crop_list

#================================= yolov3_helper.py======================================
MIN_CONF = 0.5
NMS_THRESH = 0.3

DEFAULT_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.3

LABELS = open(LABEL_PATH).read().strip().split('\n')
COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype = 'uint8')

DIR_PATH = ''

model = 'output/models/inference/yolov3-custom_last.weights'
model_config = 'cfg/yolov3-custom.cfg'
labels = 'cfg/obj.names'
# input_videos = 'videos/'
# output_video = 'output/output_video.mp4'

MODEL_PATH = model
CONFIG_PATH = DIR_PATH + model_config
LABEL_PATH = DIR_PATH + labels

crop, image = None, None

# Initialization
# load the COCO class labels our YOLO model was trained on

# derive the paths to the YOLO weights and model configuration
weightsPath = MODEL_PATH
configPath = CONFIG_PATH

def yolo_detector(frame, net, ln, MIN_CONF, Idx=0):
    # grab the dimensions of the frame and  initialize the list of
    # results
    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()

    layerOutputs = net.forward(ln)
    st.sidebar.success("Processing time for YOLOV3: {} {:.4f} seconds.".format('\n',time.time() - start))
    print("Processing time for YOLOV3: --- {:.4f} seconds ---".format(time.time() - start))
    # initialize our lists of detected bounding boxes, centroids, and
    # confidences, respectively
    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter detections by (1) ensuring that the object
            # detected was a person and (2) that the minimum
            # confidence is met
            if classID == Idx and confidence > MIN_CONF:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_network(configpath, weightspath):

    with st.spinner("Loading Yolo weights!"):
        # load our YOLO object detector trained on our dataset (1 class)
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # determine only the *output* layer names that we need from YOLO
    output_layer_names = net.getLayerNames()
    output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layer_names

def yolo_crop_correction(frame, bbox, w, h):
    # resizing cropped image
    (startX, startY, endX, endY) = bbox
    
    crop = cropped_image(frame, (startX, startY, endX, endY))

    crop_w, crop_h = endX - startX, endY - startY # height & width of number plate of 416*416 image
    width_m, height_m = w/416, h/416 # width and height multiplier
    w2, h2 = round(crop_w*width_m), round(crop_h*height_m)
    crop = cv2.resize(np.asarray(crop), (w2, h2))
    return crop
    
def yolo_inference(image, confidence_cutoff):
    # YOLO Detection
    # Preprocess
    frame = cv2.resize(np.asarray(image), (416, 416))

    # Get parameter
    MIN_CONF = confidence_cutoff
    w, h = image.size

    net, output_layer_names = load_network(configPath, weightsPath)
    results = yolo_detector(frame, net, output_layer_names, MIN_CONF, Idx=LABELS.index("number_plate"))

    crop_list = []
    # Loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # Extract the bounding box and centroid coordinates
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid

        # crop correct and multiple image cropping
        try:
            crop = yolo_crop_correction(frame, bbox, w, h)
            crop_list.append([crop, prob])
            
        except NameError as e:
            st.error('''
            Model is not confident enough!
            \nTry lowering the confidence cutoff score from sidebar.
            ''')
            st.error("Error log: "+str(e))

        # Overlay
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
        # cv2.circle(frame, (cX, cY), 5, (0, 255, 0), 1)

    # Show result
    image = cv2.resize(np.asarray(frame), (w, h)) # resizing image as yolov3 gives 416*416 as output
    
    return image, crop_list

#============================ About ==========================
def about():

    st.warning("""
    ## \u26C5 Behind The Scenes
        """)
    st.success("""
    To see how it works, please click the button below!
        """)
    github = st.button("👉🏼 Click Here To See How It Works")
    if github:
        github_link = "https://github.com/udaylunawat/Automatic-License-Plate-Recognition"
        try:
            webbrowser.open(github_link)
        except:
            st.error("""
                ⭕ Something Went Wrong!!! Please Try Again Later!!!
                """)
    st.info("Built with Streamlit by [Uday Lunawat 😎](https://github.com/udaylunawat)")


# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def about_yolo():
    yolo_dir = 'banners/yolo/'
    yolo_banner = random.choice(listdir(yolo_dir))
    st.sidebar.image(yolo_dir+yolo_banner, use_column_width=True)
    
    st.sidebar.info(
        "**YOLO (“You Only Look Once”)** is an effective real-time object recognition algorithm, \
        first described in the seminal 2015 [**paper**](https://arxiv.org/abs/1506.02640) by Joseph Redmon et al.\
        It's a network that uses **Deep Learning (DL)** algorithms for **object detection**. \
        \n\n[**YOLO**](https://missinglink.ai/guides/computer-vision/yolo-deep-learning-dont-think-twice/) performs object detection \
        by classifying certain objects within the image and **determining where they are located** on it.\
        \n\nFor example, if you input an image of a herd of sheep into a YOLO network, it will generate an output of a vector of bounding boxes\
            for each individual sheep and classify it as such. Yolo is based on algorithms based on regression━they **scan the whole image** and make predictions to **localize**, \
        identify and classify objects within the image. \
        \n\nAlgorithms in this group are faster and can be used for **real-time** object detection.\
        **YOLO V3** is an **improvement** over previous YOLO detection networks. \
        Compared to prior versions, it features **multi-scale detection**, stronger feature extractor network, and some changes in the loss function.\
        As a result, this network can now **detect many more targets from big to small**. \
        And, of course, just like other **single-shot detectors**, \
        YOLO V3 also runs **quite fast** and makes **real-time inference** possible on **GPU** devices.")


# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def about_retinanet():
    od_dir = 'banners/Object detection/'
    od_banner = random.choice(listdir(od_dir))
    st.sidebar.image(od_dir+od_banner, use_column_width=True)

    st.sidebar.info(
        "[RetinaNet](https://arxiv.org/abs/1708.02002) is **one of the best one-stage object detection models** that has proven to work well with dense and small scale objects. \
        For this reason, it has become a **popular** object detection model to be used with aerial and satellite imagery. \
        \n\n[RetinaNet architecture](https://www.mantralabsglobal.com/blog/better-dense-shape-detection-in-live-imagery-with-retinanet/) was published by **Facebook AI Research (FAIR)** and uses Feature Pyramid Network (FPN) with ResNet. \
        This architecture demonstrates **higher accuracy** in situations where *speed is not really important*. \
        RetinaNet is built on top of FPN using ResNet.")


#================================= Functions =================================

def streamlit_preview_image(image):
    st.sidebar.image(
                image,
                use_column_width=True,
                caption = "Original Image")

def streamlit_output_image(image, caption):
    st.image(image,
            use_column_width=True,
            caption = caption)

def streamlit_OCR(output_image):

    st.write("## 🎅 Bonus:- Optical Character Recognition (OCR)")
    
    st.error("Note: OCR is performed on the enhanced cropped images.")
    
    note = st.empty()
    OCR_type = st.radio("OCR Mode",["Google's Tesseract OCR","easy_OCR","Secret Combo All-out Attack!!"])
    button = st.empty()
    ocr_text = st.empty()
    if button.button('Recognize Characters !!'):
        
        if OCR_type == "Google's Tesseract OCR":
            # try:
            tessy_ocr = OCR(output_image)
            
            if len(tessy_ocr) == 0:
                ocr_text.error("Google's Tesseract OCR Failed! :sob:")

            else:
                ocr_text.success("Google's Tesseract OCR: " + tessy_ocr)
                st.error("Researching Google's Tesseract OCR is a work in progress 🚧\
            \nThe results might be unreliable.")

        elif OCR_type == "easy_OCR":

            try:
                easy_ocr = easy_OCR(output_image)
                ocr_text.success("easy OCR: " + easy_ocr)

            except NameError:
                ocr_text.error("EasyOCR not installed")

            except ModuleNotFoundError:
                ocr_text.error("EasyOCR not installed")

            except:
                ocr_text.error("Easy OCR Failed! :sob:")

        elif OCR_type == "Secret Combo All-out Attack!!":

            try_all_OCR(output_image)

def multi_crop(image, crop_list):

    if len(crop_list)!=0: 
        # https://dbader.org/blog/python-min-max-and-nested-lists
        [max_crop, max_conf] = max(crop_list, key=lambda x: x[1])

        st.write("## License Plate Detection!")
        streamlit_output_image(image, 'Annotated Image with model confidence score: {0:.2f}'.format(max_conf))
        
        img_list, score_list =  map(list, zip(*crop_list))
        st.write("### Cropped Plates")
        st.image(img_list, caption=["Cropped Image with model confidence score:"+'{0:.2f}'.format(score) for score in score_list], width = crop_size)
    else:
        st.error("Plate not found! Reduce confidence cutoff or select different image.")
        return
    return max_crop

#======================== Time To See The Magic ===========================

st.sidebar.markdown("## Automatic License Plate recognition system 🇮🇳")
st.sidebar.markdown("Made with :heart: in India by [Uday Lunawat](https://udaylunawat.github.io)")

crop, image = None, None
img_size, crop_size = 600, 400

activities = ["Home", "YoloV3 Detection", "RetinaNet Detection", "About"]
choice = st.sidebar.radio("Go to", activities)

if choice == "Home":
    
    st.markdown("<h1 style='text-align: center; color: black;'>Indian ALPR System using Deep Learning 👁</h1>", unsafe_allow_html=True)
    st.sidebar.info(__doc__)
    st.write("## How does it work?")
    st.write("Add an image of a car and a [deep learning](http://wiki.fast.ai/index.php/Lesson_1_Notes) model will look at it\
         and find the **license plate** like the example below:")
    st.sidebar.info("- The learning (detection) happens  \
                    with a fine-tuned [**Retinanet**](https://arxiv.org/abs/1708.02002) or a [**YoloV3**](https://pjreddie.com/darknet/yolo/) \
                    model ([**Google's Tensorflow 2**](https://www.tensorflow.org/)), \
                    \n- This front end (what you're reading) is built with [**Streamlit**](https://www.streamlit.io/) \
                    \n- It's all hosted on the cloud using [**Google Cloud Platform's App Engine**](https://cloud.google.com/appengine/).")
                    
    # st.video("https://youtu.be/C_lIenSJb3c")
    #  and a [YouTube playlist](https://www.youtube.com/playlist?list=PL6vjgQ2-qJFeMrZ0sBjmnUBZNX9xaqKuM) detailing more below.")
    # or OpenCV Haar cascade

    st.sidebar.warning("#### Checkout the Source code on [GitHub](https://github.com/udaylunawat/Automatic-License-Plate-Recognition)")

    st.image("output/sample_output.png",
            caption="Example of a model being run on a car.",
            use_column_width=True)

    st.write("## How is this made?")
    banners = 'banners/'
    files = [banners+f for f in os.listdir(banners) if os.path.isfile(os.path.join(banners, f))]
    st.image(random.choice(files),use_column_width=True)


elif choice == "About":
    about()

elif choice == "RetinaNet Detection" or "YoloV3 Detection":

    crop, image = None, None

    st.write("## Upload your own image")

    # placeholders
    choose = st.empty() 
    upload = st.empty()
    note = st.empty()
    note.error("**Note:** The model has been trained on Indian cars and number plates, and therefore will only work with those kind of images.")

    # Detections below this confidence will be ignored
    confidence = st.slider("Confidence Cutoff",0.0,1.0,(0.5))
    predictor = st.checkbox("Make a Prediction 🔥")

    samplefiles = sorted([sample for sample in listdir('data/sample_images')])
    radio_list = ['Choose existing', 'Upload']

    query_params = st.experimental_get_query_params()
    # Query parameters are returned as a list to support multiselect.
    # Get the second item (upload) in the list if the query parameter exists.
    # Setting default page as Upload page, checkout the url too. The page state can be shared now!
    default = 1

    activity = choose.radio("Choose existing sample or try your own:", radio_list, index=default)
    
    if activity:
        st.experimental_set_query_params(activity=radio_list.index(activity))
        if activity == 'Choose existing':
            selected_sample = upload.selectbox("Pick from existing samples", (samplefiles))
            image = Image.open('data/sample_images/'+selected_sample)
            IMAGE_PATH = 'data/sample_images/'+selected_sample
            image = Image.open('data/sample_images/'+selected_sample)
            img_file_buffer = None

        else:
            # You can specify more file types below if you want
            img_file_buffer = upload.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'], multiple_files = True)

            IMAGE_PATH = img_file_buffer
            try:
                image = Image.open(IMAGE_PATH)
            except:
                pass

            selected_sample = None

    if image:
        
        st.sidebar.markdown("## Preview Of Selected Image! 👀")
        streamlit_preview_image(image)
        
        docs = st.sidebar.empty()
        if docs.checkbox("View Documentation"):
            if choice == "RetinaNet Detection":
                about_retinanet()
            else:
                about_yolo()

    else :

        if choice == "RetinaNet Detection":
            about_retinanet()
        else:
            about_yolo()

    max_crop, max_conf = None, None

    if choice == "RetinaNet Detection":

        if image:
            if predictor:
                model = load_retinanet()
                try:
                    # gif_runner = image_holder.image('banners/processing.gif', width = img_size)
                    annotated_image, score, crop_list = retinanet_detector(IMAGE_PATH, model, confidence)
                    # gif_runner.empty()
                    max_crop = multi_crop(annotated_image, crop_list)

                except TypeError as e:

                    st.warning('''
                            Model is not confident enough!
                            \nTry lowering the confidence cutoff.
                            ''')
                    # st.error("Error log: "+str(e))

                if max_crop!= None:
                    enhanced = enhance_crop(max_crop)
                    streamlit_OCR(enhanced)

    if choice == "YoloV3 Detection":
        if image:
            if predictor:
                try:
                    # gif_runner = image_holder.image('banners/processing.gif', width = img_size)
                    image, crop_list = yolo_inference(image, confidence)
                    # gif_runner.empty()
                    max_crop = multi_crop(image, crop_list)
                    
                except UnboundLocalError as e:
                    st.write(e)

                except NameError as e:
                    st.error('''
                    Model is not confident enough!
                    \nTry lowering the confidence cutoff.
                    ''')
                    st.error("Error log: "+str(e))
                
                if isinstance(max_crop, np.ndarray):
                    max_crop = Image.fromarray(max_crop)
                    enhanced = enhance_crop(max_crop)
                    streamlit_OCR(enhanced)

