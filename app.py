import numpy as np
from flask import Flask, request, render_template
import webbrowser
from tensorflow import keras
import cv2
import base64
import os
from werkzeug.utils import secure_filename

model = keras.models.load_model(r'C:\Users\Asmae ER\Desktop\hcr\model_hand.h5')
# Dictionary for getting characters from index values...
word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
             13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}



# Prediction on external image...
def pi(img):
  print("pi***************************************")
  #img = cv2.imread(r'F:\shared\hcr\19740085.jpg')
  #img = cv2.imread(k)
  img_copy = img.copy()

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (400, 440))

  img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
  img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
  _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

  img_final = cv2.resize(img_thresh, (28, 28))
  img_final = np.reshape(img_final, (1, 28, 28, 1))


  img_pred = word_dict[np.argmax(model.predict(img_final))]

  cv2.putText(img, "Dataflair _ _ _ ", (20, 25),
            cv2.FONT_HERSHEY_TRIPLEX, 0.7, color=(0, 0, 230))
  cv2.putText(img, "Prediction: " + img_pred, (20, 410),
            cv2.FONT_HERSHEY_DUPLEX, 1.3, color=(255, 0, 30))
  cv2.imshow('Dataflair handwritten character recognition _ _ _ ', img)
  return img_pred
'''
img2 =cv2.imread(r'F:\shared\hcr\19740085.jpg')
print(pi(img2))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2, (400, 440))
cv2.imshow('img2',img2)
'''


app = Flask(__name__)
#model = pickle.load(open('modelSVM.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index.html',photo="static/upload_img.jpg")

@app.route('/upload', methods=['POST'])
def upload():
  print("up***************************************")
  '''
  print("up*****************************************************")
  imagefile = request.files['imagefile']
  #npimg = np.fromstring(imagefile, np.uint8)
  npimg = np.fromfile(imagefile, np.uint8)
 # convert numpy array to image
  img = cv2.imdecode(npimg, cv2.COLOR_BGR2RGB)
  #nparr = np.fromstring(imagefile, np.uint8)
  #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  #print(imagefile)
  cv2.imshow("OpenCV Image Reading", img)
  print("after print*****************************************************")
  '''
  '''
  #read image file string data
  filestr = request.files['imagefile'].read()
  #convert string data to numpy array
  npimg = np.fromstring(filestr, np.uint8)
  # convert numpy array to image
  img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
'''
  '''
  #bimg=request.form['bimg']
  imagefile=request.files['image']
  npimg = np.fromstring(imagefile, np.uint8)
  npimg = np.fromfile(imagefile, np.uint8)
  # convert numpy array to image
  img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
  pre=pi(img)
  print(pre)
  print("pi***************************************")
  print(request.form['bimg'])
  cv2.destroyAllWindows()
  return render_template('index.html' , prediction_text='detected character {}'.format(pre),photo=bimg)
  #return render_template('index.html')
      '''
  UPLOAD_FOLDER = (r'C:\Users\Asmae ER\Desktop\hcr\static')
  app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
  image =request.files['image']
  filename = secure_filename(image.filename) # save file 
  filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  filepathh="static/"+filename
  print(filepathh)
  image.save(filepath)
  #npimg = np.fromfile(image, np.uint8)
  #img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
  img =cv2.imread(filepath)
  pre=pi(img)
  print(pre)
  print("pi***************************************")
  cv2.destroyAllWindows()

  return render_template('index.html' , prediction_text='detected character {}'.format(pre),photo=filepathh)

  '''
  UPLOAD_FOLDER = 'F:\shared\hcr\static'
  app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
  data =request.files['imagefile']
  filename = secure_filename(data.filename) # save file 
  filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  data.save(filepath)
  img2 =cv2.imread(r'F:\shared\hcr\19740085.jpg')
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
  img2 = cv2.resize(img2, (400, 440))
  cv2.imshow('img2',img2)
  print("*******************************************************")
  print(filepath)
  img=cv2.imread(filepath)
  cv2.imshow("OpenCV Image Reading", img)
  #print(pi(img))
  return render_template('index3.html')
  '''



  '''
@app.route('/predict',methods=['POST'])
def predict():
  input_features = [int(x) for x in request.form.values()]
  print(input_features)
  features_value = [np.array(input_features)]

  features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli']
  '''
  '''
  df = pd.DataFrame(features_value)
  output = model.predict(df)

  if output == 4:
      res_val = "tumeur  maligne (cancereuse)"
  else:
      res_val = "tumeur bénigne (non cancereuse)"

  '''


  #return render_template('index.html', prediction_text='Patient a une {}'.format(res_val))

if __name__ == "__main__":
  webbrowser.open('http://127.0.0.1:5000', new=2)
  app.run()