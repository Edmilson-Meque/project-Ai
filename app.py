from flask import Flask, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import sqlite3 # Importar a biblioteca da base de dados
import atexit # Para fechar a base de dados ao sair

app = Flask(__name__)

# --- Configuração da Base de Dados ---
DB_NAME = 'health_data.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Criamos a tabela se ela não existir
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS health_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        heart_rate INTEGER,
        blood_oxygen INTEGER,
        status TEXT,
        recommendation TEXT
    )
    ''')
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.text_factory = str  # <-- Verifique se esta linha AINDA ESTÁ AQUI
    return conn
# Executar init_db na inicialização
init_db()

# --- Carregamento do Modelo de IA ---
try:
    model = joblib.load('model.joblib')
    print("--- Modelo 'model.joblib' carregado com sucesso. ---")
except FileNotFoundError:
    print("!!! Erro: Ficheiro 'model.joblib' não encontrado.")
    print("!!! Por favor, execute 'python model.py' primeiro para treinar o modelo.")
    exit()

def get_real_time_data():
    """Simula uma única e nova leitura do dispositivo wearable."""
    if np.random.rand() > 0.1: 
        heart_rate = np.random.randint(65, 95)
        blood_oxygen = np.random.randint(95, 100)
    else:
        if np.random.rand() > 0.5:
            heart_rate = np.random.choice([np.random.randint(45, 55), np.random.randint(105, 120)])
            blood_oxygen = np.random.randint(95, 100)
        else:
            heart_rate = np.random.randint(65, 95)
            blood_oxygen = np.random.randint(85, 93)
    return {'heart_rate': heart_rate, 'blood_oxygen': blood_oxygen}

@app.route('/')
def home():
    # 1. Obter "novos" dados do wearable
    current_data = get_real_time_data()
    
    # 2. Preparar os dados para o modelo
    data_point = pd.DataFrame([current_data])
    features = data_point[['heart_rate', 'blood_oxygen']]

    # 3. Fazer a previsão (1 = normal, -1 = anomalia)
    prediction = model.predict(features)
    
    # 4. Criar a recomendação
    status = 'Normal'
    recommendation = 'Your vital signs are stable. Keep up your healthy habits.'
    
    if prediction[0] == -1: # Anomalia Detectada!
        status = 'Anomaly Detected!'
        if current_data['blood_oxygen'] < 94:
            recommendation = 'Blood oxygen level is low ({}%). Please rest and take deep breaths.'.format(current_data['blood_oxygen'])
        elif current_data['heart_rate'] < 60 or current_data['heart_rate'] > 100:
            recommendation = 'Irregular heart rate detected ({} bpm). Try to relax.'.format(current_data['heart_rate'])
        else:
             recommendation = 'We detected an unusual reading. Monitor your symptoms and rest.'

    # 5. GUARDAR os dados na base de dados
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO health_readings (heart_rate, blood_oxygen, status, recommendation) VALUES (?, ?, ?, ?)",
        (current_data['heart_rate'], current_data['blood_oxygen'], status, recommendation)
    )
    conn.commit()
    conn.close()

    # 6. LER o histórico da base de dados (últimos 20)
    conn = get_db_connection()
    readings_from_db = conn.execute(
        "SELECT timestamp, heart_rate, blood_oxygen FROM health_readings ORDER BY timestamp DESC LIMIT 20"
    ).fetchall()
    conn.close()

    # Preparar dados para o gráfico (precisamos de reverter a ordem para o gráfico ficar cronológico)
    history = {
        'labels': [r['timestamp'].split(' ')[1] for r in reversed(readings_from_db)], # Apenas a hora
        'heart_rates': [r['heart_rate'] for r in reversed(readings_from_db)],
        'blood_oxygens': [r['blood_oxygen'] for r in reversed(readings_from_db)]
    }

    # 7. Enviar todos os dados para a página web
    health_data = {
        'heart_rate': current_data['heart_rate'],
        'blood_oxygen': current_data['blood_oxygen'],
        'status': status,
        'recommendation': recommendation
    }
    
    # Passamos os dados actuais E o histórico
    return render_template('index.html', data=health_data, history=history)

if __name__ == '__main__':
    app.run(debug=True)