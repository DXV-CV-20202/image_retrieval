from re import L
from flask import Flask, render_template, request, send_from_directory
import os, argparse
from retrieve import MatchApp

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

    query_results = match_app.do_match(save_path)

    results_path = list(map(lambda x: x[1]['image_path'], query_results))
    return render_template("index.html", original=save_path[7:], results=results_path)


@app.route('/image/<path:path>')
def get_image(path):
    return send_from_directory('.', path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default='localhost',
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=5000,
        help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    match_app = MatchApp()

    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)