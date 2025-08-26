import inspect
import os
# from argparse import ArgumentParser

from flask import Flask, request
from flask_cors import CORS

from prediction import extract_entities, load_model

# parser = ArgumentParser()
# parser.add_argument("--port", type=int, default=25000)
# parser.add_argument("--device", type=str, default="cpu")
# args = parser.parse_args()

DEVICE = os.environ.get("DEVICE", "cpu")
PORT = int(os.environ.get("PORT", 25000))

app = Flask(__name__)
CORS(app)

load_model(DEVICE)


def validate_args(func):
    try:
        example = request.json
    except Exception:
        raise ValueError("Invalid JSON request.")
    if "text" not in example:
        raise ValueError("Missing a 'text' key in the request.")
    expected_args = set(inspect.getfullargspec(func)[0])
    requested_args = set(example.keys())
    warning = None
    if not requested_args.issubset(expected_args):
        msg = ", ".join(requested_args - expected_args)
        warning = f"Ignored invalid keys in the request: {msg}"
        for key in requested_args - expected_args:
            del example[key]
    return warning, example


@app.route("/", methods=["POST"])
def ner():
    try:
        warning, example = validate_args(extract_entities)
    except ValueError as e:
        return {"error": str(e)}, 400
    return {
        **extract_entities(**example),
        "warning": warning,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
