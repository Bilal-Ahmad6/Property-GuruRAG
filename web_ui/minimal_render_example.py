import os

port = int(os.environ.get("PORT", 5000))  # default to 5000 if PORT not set

from flask import Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Render!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
