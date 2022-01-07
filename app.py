
import streamlit as st
import cv2 
import numpy as np
import Inference_Math_Detection as MD
import Recog_MathForm as RM
from PIL import Image
import pdf2image
import os
from .session_state import get_session_state
def download_models():
    mathdetector = './Models/MathDetector.ts'
    mathrecog = './Models/MathRecog.pth'
    
    if not os.path.exists(mathdetector):
        detector_url = 'gdown -O '+mathdetector+' https://drive.google.com/uc?id=1AGZTIRbx-KmLQ7bSEAcxUWWtdSrYucFz'
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(detector_url)
    else:
        print("Detector Model is here")

    if not os.path.exists(mathrecog):
        detector_url = 'gdown -O '+mathrecog+' https://drive.google.com/uc?id=1oR7eNBOC_3TBhFQ1KTzuWSl7-fet4cYh'
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(detector_url)
    else:
        print("Reconizer Model is here")

def draw_rectangles (image, preds):
    for each_pred in preds:
        cv2.rectangle(image, (int(each_pred[0]),int(each_pred[1])), (int(each_pred[2]),int(each_pred[3])),(255,0,0),2)

download_models()
math_model = MD.initialize_model("./Models/MathDetector.ts")
mathargs, *mathobjs = RM.initialize()

inf_style = st.sidebar.selectbox("Inference Type",('Image', 'PDF'))
if inf_style == 'Image':
    state = get_session_state()
    if not state.widget_key:
        state.widget_key = str(randint(1000, 100000000))
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png','jpeg', 'jpg'])
    res = st.sidebar.radio("Final Result",("Detection","Detection And Recogntion"))
    if uploaded_file is not None:
        with st.spinner(text='In progress'):
            st.sidebar.image(uploaded_file)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            st.title('Mathematical Formula Detector!')
            if st.button('Launch the Detection!'):
                results_boxes = MD.predict_formulas(opencv_image,math_model)
                images_rectangles = cv2.imdecode(file_bytes, 1)
                draw_rectangles(images_rectangles,results_boxes)
                st.image(images_rectangles)
                if st.button('clear uploaded_file'):
                    state.widget_key = str(randint(1000, 100000000))
                    st.write("attempt to clear uploaded_file")
                state.sync()
                col1, col2, col3 = st.columns(3)
                col1.header("Image")
                col2.header("Latext")
                col3.header("Formula")
                if res == "Detection And Recogntion":
                    for each_box in results_boxes:
                        each_box = list(map(int,each_box))
                        crop_box = opencv_image[each_box[1]:each_box[3],each_box[0]:each_box[2],:]
                        crop_img = Image.fromarray(np.uint8(crop_box))
                        pred = RM.call_model(mathargs, *mathobjs, img=crop_img)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(crop_box)
                        with col2:
                            st.write(pred, width=5)
                        with col3:
                            st.markdown("$$"+pred+"$$")
elif inf_style == 'PDF':
    imagem_referencia = st.sidebar.file_uploader("Choose an image", type=["pdf"])
    res = st.sidebar.radio("Final Result",("Detection","Detection And Recogntion"))

    if imagem_referencia is not None:

        if imagem_referencia.type == "application/pdf":
            images = pdf2image.convert_from_bytes(imagem_referencia.read())
            page_idx = st.sidebar.number_input("Page Number", min_value=1, max_value=len(images), value=1, step=1)
            opencv_image = np.array(images[int(page_idx)-1])
            results_boxes = MD.predict_formulas(opencv_image,math_model)
            images_rectangles = np.array(images[int(page_idx)-1])
            draw_rectangles(images_rectangles,results_boxes)
            st.image(images_rectangles)
            col1, col2, col3 = st.columns(3)
            col1.header("Image")
            col2.header("Latext")
            col3.header("Formula")
            if res == "Detection And Recogntion":
                for each_box in results_boxes:
                    each_box = list(map(int,each_box))
                    crop_box = opencv_image[each_box[1]:each_box[3],each_box[0]:each_box[2],:]
                    crop_img = Image.fromarray(np.uint8(crop_box))
                    pred = RM.call_model(mathargs, *mathobjs, img=crop_img)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(crop_box)
                    with col2:
                        st.markdown(pred)
                    with col3:
                        st.markdown("$$"+pred+"$$")

    
