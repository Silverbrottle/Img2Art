import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
from load import style_transfer

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg', '.jfif']
app.config['UPLOAD_PATH'] = 'user_input'
save_dir = 'output'


@app.route('/')
def index():
    return render_template('indexnew.html')


@app.route('/style', methods=['GET', 'POST'])
def style():

    if request.method == 'POST' and 'fileform' in request.form:
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            filename = 'USER_STYLE_TRANSFER '+filename
            uploaded_file.save(os.path.join(
                app.config['UPLOAD_PATH'], filename))
            files = os.listdir(app.config['UPLOAD_PATH'])[-1]
            print(files)
            return render_template('style_transfer.html', files=files)

    elif request.method == 'POST':
        files = os.listdir(app.config['UPLOAD_PATH'])
        style_filename = files[0]
        file_path = os.path.join(app.config['UPLOAD_PATH'], style_filename)
        print(file_path)
        stylename = request.form['artist']
        style_transfer(file_path, stylename)
        files = os.listdir(save_dir)[-1]
        return render_template('style_transfer.html', files=files)

    else:
        filelist = [f for f in os.listdir(save_dir)]
        filelist2 = [f for f in os.listdir(app.config['UPLOAD_PATH'])]
        print("Clearing filelist1...")
        for f in filelist:
            print(f)
            os.remove(os.path.join(save_dir, f))
        print("Clearing filelist2...")
        for f in filelist2:
            print(f)
            os.remove(os.path.join(app.config['UPLOAD_PATH'], f))
        return render_template('style_transfer.html')


@app.route('/uploads/<filename>')
def upload(filename):
    sub_str = 'USER_STYLE_TRANSFER'
    if (sub_str in filename):
        return send_from_directory(app.config['UPLOAD_PATH'], filename)
    else:
        return send_from_directory(save_dir, filename)


@app.route('/<artist_name>')
def artists(artist_name):
    if artist_name == 'vangogh':
        return render_template('vangogh.html')
    if artist_name == 'paulcezzane':
        return render_template('paulcezzane.html')
    if artist_name == 'berthemorisot':
        return render_template('berthemorisot.html')
    if artist_name == 'jacksonpollock':
        return render_template('jacksonpollock.html')
    if artist_name == 'monet':
        return render_template('monet.html')
    if artist_name == 'pablopicasso':
        return render_template('pablopicasso.html')
    if artist_name == 'paulgaguin':
        return render_template('paulgaguin.html')
    if artist_name == 'peploe':
        return render_template('peploe.html')
    if artist_name == 'wassilyk':
        return render_template('wassilyk.html')
    if artist_name == 'edvardmunich':
        return render_template('edvardmunich.html')


if __name__ == '__main__':
    app.run(debug=True)
