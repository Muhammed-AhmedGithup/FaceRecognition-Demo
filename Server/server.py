from flask import Flask, request, jsonify
import utils
app = Flask(__name__)


@app.route('/')
def index():
    return 'hi'

@app.route('/classify_image',methods=['GET','POST'])
def classify_image():
    image_data=request.form['image_data']
    print(type(image_data))
    print(utils.classify_image(image_data))
    response=jsonify(utils.classify_image(image_data))
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response





if __name__ == '__main__':
    utils.load_artifactes()
    app.run(debug=True,port=5000)