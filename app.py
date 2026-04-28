from flask import Flask, request, render_template_string
import os
from predict import predict_genre

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HTML = """
<!doctype html>
<html>
<head>
<title>Genre AI</title>
</head>

<body style="background:#0b0f14; color:white; font-family:sans-serif;">

<div style="max-width:600px; margin:80px auto;">

<h1>Genre AI</h1>
<p style="color:#aaa;">Upload a track and get its genre</p>

<form method="post" enctype="multipart/form-data">

<input type="file" name="file" style="margin-top:20px;"><br><br>

<button type="submit"
style="padding:10px 20px; background:#00ffc3; border:none; color:black;">
Analyze
</button>

</form>

{% if result %}
<p style="margin-top:20px; color:#00ffc3; font-size:20px;">
{{ result }}
</p>
{% endif %}

{% if error %}
<p style="color:red;">{{ error }}</p>
{% endif %}

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    temp_path = None

    if request.method == "POST":
        try:
            file = request.files.get("file")

            if not file or file.filename == "":
                raise Exception("No file selected")

            if not file.filename.lower().endswith(".wav"):
                raise Exception("Only .wav files supported")

            temp_path = os.path.join(BASE_DIR, "temp.wav")
            file.save(temp_path)

            result = predict_genre(temp_path)

        except Exception as e:
            error = str(e)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return render_template_string(HTML, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)