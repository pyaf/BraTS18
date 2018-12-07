import os
import sys
import time
import pdb
from flask import Flask, render_template, request, Response, jsonify, send_file
from threading import Thread, active_count
from test import Model
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
model = Model(log=app.logger.info)


@app.route("/")
def index():
    return render_template("index.html")


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


@app.route("/postScans", methods=["POST"])
def postScans():
    """Recieves Scans, saves them in files/scans/ directory"""
    try:
        print("Recieved scan")
        file = request.files["scans"]
        filename = file.filename
        print(file, filename)
        patient_id = request.args["patient_id"]
        print(patient_id)
        scanfolder = os.path.join("files/scans/", patient_id)
        mkdir(scanfolder)
        mkdir(os.path.join("files/segmentations/", patient_id))
        scanpath = os.path.join(scanfolder, filename)
        file.save(scanpath)
        Thread(
            target=model.segment, args=(scanpath,)
        ).start()  # run in a thread, takes 6 mins
        return jsonify(success=True)
    except Exception as e:
        return jsonify(response=str(e), success=False)


@app.route("/scanStatus", methods=["POST"])
def SegStatus():
    """Check if segmentation is ready or not, return if ready"""
    try:
        patient_id = request.args["patient_id"]
        scan_id = request.args["scan_id"]
        seg_file = os.path.join(
            "files/segmentations/", patient_id, "%s.nii.gz" % scan_id
        )
        app.logger.info("Segmentation ready, returning..")
        return send_file(seg_file)
    except Exception as e:
        app.logger.info("Segmentation not available..")
        return Response(status=404)


if __name__ == "__main__":
    # handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
    # handler.setLevel(logging.INFO)
    # app.logger.addHandler(handler)
    # app.run()
    app.run(host="0.0.0.0", port=6006, threaded=True)
