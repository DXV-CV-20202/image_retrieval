from re import L
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os, sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@app.route("/")
def index():
    return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/search", methods=['POST'])
def search():
    print(request.files)
    if 'q' not in request.files:
        error = 'No file uploaded.'
        return render_template("index.html", error=error)


    f = request.files['q']

    if f.filename == '' or not allowed_file(f.filename):
        error = 'Invalid file type.'
        return render_template("index.html", error=error)

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(save_path)
    app.logger.info('Saved to: ' + save_path)

    # Some stuff here

    query_results = [
        "img/0001.jpg",
        "img/0002.jpg",
        "img/0003.jpg",
        "img/0004.jpg",
    ]

    return render_template("index.html", results=query_results, original=save_path[7:])


# @app.route("img")
# def get_image():
#     return send_from_directory(filename, mimetype='image/gif')