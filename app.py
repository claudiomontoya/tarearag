import uuid
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import PyPDF2
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import re
app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

QDRANT_HOST = os.environ.get('QDRANT_HOST')
QDRANT_PORT = int(os.environ.get('QDRANT_PORT', 6333))
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

# Inicializar clientes
qdrant_client = QdrantClient(
    url=f"https://{QDRANT_HOST}:{QDRANT_PORT}",
    api_key=QDRANT_API_KEY
)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Asegurarse de que la colección existe
collection_name = "pdf_collection"
try:
    qdrant_client.get_collection(collection_name)
except Exception as e:
    print(f"La colección {collection_name} no existe. Creándola...")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
    )
    print(f"Colección {collection_name} creada exitosamente.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text, max_chunk_size=7500):
    # Dividir el texto en párrafos utilizando saltos de línea dobles o simples como delimitadores
    chunks = re.split(r'\n\s*\n', text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip() != '']
    
    sub_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            sub_chunks.extend([chunk[i:i + max_chunk_size] for i in range(0, len(chunk), max_chunk_size)])
        else:
            sub_chunks.append(chunk)
    return sub_chunks


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Procesar el archivo en memoria
            file_stream = BytesIO(file.read())
            
            # Extraer texto del PDF en memoria
            text = extract_text_from_pdf(file_stream)
            chunks = split_text_into_chunks(text)

            for index, chunk in enumerate(chunks):
                try:
                    vector = encoder.encode(chunk).tolist()
                    point_id = str(uuid.uuid4())
                    
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=[
                            {
                                "id": point_id,
                                "vector": vector,
                                "payload": {"text": chunk, "filename": filename, "chunk_index": index + 1}
                            }
                        ]
                    )
                except Exception as e:
                    print(f"Error al procesar el chunk {index + 1}: {str(e)}")
                    return jsonify({"error": str(e)}), 500
            
            return jsonify({"message": "File uploaded and indexed successfully", "chunks_processed": len(chunks)}), 200
        return jsonify({"error": "File type not allowed"}), 400
    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

def extract_text_from_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
    
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        if 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        question_vector = encoder.encode(question).tolist()
        
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=question_vector,
            limit=1
        )
        
        if not search_result:
            return jsonify({"error": "No relevant documents found"}), 404
        
        context = search_result[0].payload['text']
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
        )
        
        answer = response.choices[0].message.content
        
        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(f"Error during query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
