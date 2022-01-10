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

Installing and running the app
================================
Clone the project repository to your machine
**In your terminal run the following commands:**
1. cd cloned-project-directory/
2. pip install -r requirements.txt
3. streamlit run app.py
A link to the app will appear in the terminal click on it : http://localhost:8501/
You will be directed to the main app page:
![1](https://user-images.githubusercontent.com/24214295/148830933-7d72121b-1064-4c68-8d1c-a681ba83b9b1.png
choose to upload an image
![22](https://user-images.githubusercontent.com/24214295/148831159-af92c059-1b49-44d1-9388-aa7d4e775f4d.png)
Click Launch Detection,the output should look like this
![22](https://user-images.githubusercontent.com/24214295/148831389-abd4ec11-ddd6-442e-a5a9-6823b32d5f92.png)
You can also choose to upload a pdf and repeat the same steps.

  
