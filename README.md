# Math Formula Detection
As large amounts of technical documents have been published in recent years, efficiently retrieving relevant documents and identifying locations of targeted terms are urgently needed.

Model
=====
We propose an end to end system that takes a picture of printed document and outputs the locations of math formulas in the image.
In our project we use a pretrained model Yolov5 as feature extractor and detector.

Data
=====
Data is collected from ICDAR competition for the 2 years 2019 and 2021.
Data can be found here: https://www.kaggle.com/ro101010/math-formula-detection

Results
=======
We managed to achieve a precision of 0.943 , a recall of 0.912 and mean average precision @ IOU = 0.5 of 0.949.

Installation 
========================
Clone the project repository to your machine
**In your terminal run the following commands:**
1.cd clone-project-directory/
2.pip install -r requirements.txt
3.streamlit run app.py
A link to the app will appear in the terminal click on it : http://localhost:8501/
You will be directed to the main app page:

  
