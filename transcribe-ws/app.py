
from flask import Flask, request, jsonify
import whisper
import os
import logging
import torch
from concurrent.futures import ThreadPoolExecutor
from TTS.api import TTS

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración global
TEMP_DIR = os.path.join(os.getcwd(), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# Inicializar modelo con configuración balanceada
def initialize_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Usando dispositivo: {device}")
    whisper_model = whisper.load_model("base").to(device)
    
    # Inicializar modelo TTS
    tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                   progress_bar=False,
                   gpu=torch.cuda.is_available())
    return whisper_model, tts_model

whisper_model, tts_model = initialize_models()
executor = ThreadPoolExecutor(max_workers=2)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo de audio'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        allowed_extensions = {'mp3', 'ogg', 'wav', 'm4a'}
        if not ('.' in audio_file.filename and audio_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Formato de archivo inválido'}), 400

        temp_path = os.path.join(TEMP_DIR, f"temp_audio_{os.urandom(8).hex()}.{audio_file.filename.rsplit('.', 1)[1].lower()}")
        
        try:
            audio_file.save(temp_path)
            logger.info(f"Archivo guardado exitosamente en {temp_path}")
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                return jsonify({'error': 'Error al guardar el archivo de audio'}), 500

            # Configuración balanceada para rendimiento y precisión
            result = whisper_model.transcribe(
                temp_path,
                fp16=torch.cuda.is_available(),  # FP16 solo si hay GPU
                language='es',
                beam_size=3,  # Reducido para mejor rendimiento
                best_of=3,    # Reducido para mejor rendimiento
                condition_on_previous_text=True,
                verbose=False  # Desactivar modo debug para mejor rendimiento
            )
            
            return jsonify({
                'success': True,
                'text': result["text"]
            })
        
        except Exception as e:
            logger.error(f"Error en la transcripción: {str(e)}")
            return jsonify({'error': f'Error en la transcripción: {str(e)}'}), 500
        
        finally:
            executor.submit(lambda: os.remove(temp_path) if os.path.exists(temp_path) else None)
    
    except Exception as e:
        logger.error(f"Error general: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No se proporcionó texto para sintetizar'}), 400

        text = data['text']
        voice = data.get('voice', 'default')  # Opcional: permitir selección de voz
        
        # Generar nombre único para el archivo de audio
        output_path = os.path.join(TEMP_DIR, f"tts_output_{os.urandom(8).hex()}.wav")
        
        try:
            # Generar audio
            tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                language='es',
                speaker_wav=voice if voice != 'default' else None
            )
            
            # Leer el archivo generado
            with open(output_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Eliminar el archivo temporal
            executor.submit(lambda: os.remove(output_path) if os.path.exists(output_path) else None)
            
            return send_file(
                io.BytesIO(audio_data),
                mimetype='audio/wav',
                as_attachment=True,
                download_name='speech.wav'
            )
            
        except Exception as e:
            logger.error(f"Error en la síntesis de voz: {str(e)}")
            return jsonify({'error': f'Error en la síntesis de voz: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error general: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Iniciando aplicación Flask...")
    app.run(debug=True, threaded=True)