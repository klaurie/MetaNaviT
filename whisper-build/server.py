from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded audio file
    audio_file = request.files['file']
    audio_path = f"/tmp/{audio_file.filename}"
    audio_file.save(audio_path)

    # Specify the model path
    model_path = "models/ggml-base.en.bin"  # Adjust as needed
    command = ["./build/bin/whisper-cli", "-m", model_path, "-f", audio_path]

    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        os.remove(audio_path)  # Clean up the temporary file
        return jsonify({"transcription": output.decode('utf-8')})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.output.decode('utf-8')}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
