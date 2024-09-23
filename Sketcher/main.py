from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from prompt_toolkit.shortcuts import message_dialog
import base64
import io
import numpy as np
import os
import cv2

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'C:/Users/Shiwangi Baranwal/Python/Flask projects/Sketcher/static/img'

class Image_processor():
    def __init__(self):
        self.image = None
        self.transformed_image = None

    def load_image(self, file):
        self.filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], self.filename)
        file.save(img_path)
        self.cv_image = cv2.imread(img_path)
        return self.cv_image
   
    def cartoon_1(self):
        if self.cv_image is None:
            return 'File not loaded'

        data = np.float32(self.cv_image).reshape((-1, 3))
        # Defining criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # Applying cv2.kmeans function
        _, label, center = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)

        # Reshape the output data to the size of input image
        result = center[label.flatten()]
        result = result.reshape(self.cv_image.shape)

        is_success, im_buf_arr = cv2.imencode(".jpg", result)
        if not is_success:
            print("Image encoding failed")
        else:
            # Convert the encoded image to a BytesIO object
            io_buf = io.BytesIO(im_buf_arr)
            # Now you can use io_buf.getvalue() to get the byte data
            byte_data = io_buf.getvalue()
        
        # Encode the byte array to base64
        self.encoded_img = base64.b64encode(byte_data).decode('utf-8')

        self.filename = secure_filename('cartoon_' + img_filename)
        return self.encoded_img
        
    def cartoon_2(self):
        if self.cv_image is None:
            return 'File not loaded'
        
        # Convert image to gray and create an edge mask using adaptive thresholding.
        gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 9)
        gray_image = cv2.medianBlur(gray_image, 3)

        # Apply a bilateral filter to smoothen the image while preserving edges.
        gray_image = cv2.bilateralFilter(self.cv_image, d=9, sigmaColor=250, sigmaSpace=100)

        # Combine the smoothened image and the edge mask to create a cartoon-like effect.
        to_cartoon = cv2.bitwise_and(gray_image, gray_image, mask=edges)
        is_success, im_buf_arr = cv2.imencode(".jpg", to_cartoon)
        
        self.encoded_img = base64.b64encode(im_buf_arr).decode('utf-8')

        self.filename = secure_filename('cartoon_' + img_filename)
        return self.encoded_img
    
    def sketch(self):
        if self.cv_image is None:
            return 'File not loaded'
        
        gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray_image)
        blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=4, sigmaY=4)
        to_sketch = cv2.divide(gray_image, 255 - blurred, scale=256.0)
        is_success, im_buf_arr = cv2.imencode(".jpg", to_sketch)
        
        self.encoded_img = base64.b64encode(im_buf_arr).decode('utf-8')
        
        self.filename = secure_filename('sketch_' + img_filename)
        return self.encoded_img

imageProcessor = Image_processor()
img_filename = ""
img = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cartoon')
def cartoon_img():
    return render_template('cartoon.html')

@app.route('/upload', methods=['POST'])
def upload():
    global img
    img = request.files['file']
    if img:
        imageProcessor.load_image(img)
        global img_filename
        img_filename = imageProcessor.filename
        
        return render_template('cartoon.html', image= img_filename)
    else:
        return "Choose a file to upload!"
    
@app.route('/cartoon1')
def cartoon1():
    if img is None:
        return "Choose an image to give it a cartoon effect!"
    
    cartoon = imageProcessor.cartoon_1()
    if cartoon is None:
        return "File not found"
    
    return render_template('cartoon.html', image= img_filename, transformed_image= cartoon)

@app.route('/cartoon2')
def cartoon2():
    if img is None:
        return "Upload an image to give it a cartoon effect!"

    cartoon = imageProcessor.cartoon_2()
    if cartoon is None:
        return "File not found"
    
    return render_template('cartoon.html', image= img_filename, transformed_image= cartoon)

@app.route('/sketch')
def sketch():
    if img is None:
        return "Upload an image to make a sketch of it!"
    
    sketch_img = imageProcessor.sketch()
    if sketch_img is None:
        return "File not found"
    
    return render_template('cartoon.html', image= img_filename, transformed_image= sketch_img)


@app.route('/download')
def download():
    if imageProcessor.encoded_img is None:
         return 'File not found'
    image_bytes = base64.b64decode(imageProcessor.encoded_img)

    # Convert the bytes to a Numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the Numpy array to an image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cartoon_path =  os.path.join(app.config['UPLOAD_FOLDER'], imageProcessor.filename)
    cv2.imwrite(cartoon_path, image)
    trans_image = imageProcessor.encoded_img
    return render_template('cartoon.html', image= img_filename, transformed_image= trans_image)

if __name__ == '__main__':
    app.run(debug=True)

