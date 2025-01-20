from flask import Flask, request, send_file
from TTS.api import TTS

app = Flask(__name__)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

@app.route("/convert", methods=["POST"])
def convert_text_to_speech():
    text = request.json.get("text")
    if not text:
        return "No text provided", 400

    output_file = "output.wav"
    tts.tts_to_file(text=text, file_path=output_file)
    return send_file(output_file, as_attachment=True)

if __name__ == "__main__":
    app.run(port=5000)