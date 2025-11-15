from flask import Flask, render_template, jsonify, request
import joblib
import numpy as np
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
import os # Para ler as variáveis de ambiente

app = Flask(__name__)

# --- Configuração da Base de Dados (PostgreSQL) ---

# 1. Lê o URL secreto da base de dados (que o Render nos vai dar)
#    Se não encontrar (porque estamos a testar localmente), usa um ficheiro sqlite.
db_url = os.environ.get('DATABASE_URL', 'sqlite:///local_test.db')

# 2. Truque para o Heroku/Render (os URLs deles começam com "postgres://")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app) # Inicia o objecto da base de dados

# --- Modelo da Base de Dados ---
# Isto define a estrutura da nossa tabela
class HealthReading(db.Model):
    __tablename__ = 'health_readings'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    heart_rate = db.Column(db.Integer)
    blood_oxygen = db.Column(db.Integer)
    status = db.Column(db.String(50))
    recommendation = db.Column(db.String(255))

# --- Cria a tabela (se não existir) ---
# Esta função corre uma vez para criar as tabelas
def init_db():
    with app.app_context():
        db.create_all()

# --- Carregamento do Modelo de IA ---
try:
    model = joblib.load('model.joblib')
    print("--- Modelo 'model.joblib' carregado com sucesso. ---")
except FileNotFoundError:
    print("!!! Erro: Ficheiro 'model.joblib' não encontrado.")
    print("!!! Por favor, execute 'python model.py' primeiro para treinar o modelo.")
    exit()

# --- Função Helper (que já tinhas) ---
def to_str(s):
    if isinstance(s, bytes):
        return s.decode('utf-8')
    return s

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

# --- Rota Principal (Atualizada com SQLAlchemy) ---
@app.route('/')
def home():
    # 1. Obter "novos" dados do wearable
    current_data = get_real_time_data()
    hr = current_data['heart_rate']
    o2 = current_data['blood_oxygen']
    
    # 2. Preparar os dados para o modelo
    data_point = pd.DataFrame([current_data])
    features = data_point[['heart_rate', 'blood_oxygen']]

    # 3. Fazer a previsão (1 = normal, -1 = anomalia)
    prediction = model.predict(features)
    
    # 4. Criar a recomendação
    status = 'Normal'
    recommendation = 'Your vital signs are stable. Keep up your healthy habits.'
    
    if prediction[0] == -1: # Anomalia
        status = 'Anomaly Detected!'
        if o2 < 94:
            recommendation = f'Blood oxygen level is low ({o2}%). Please rest and take deep breaths.'
        elif hr < 60 or hr > 100:
            recommendation = f'Irregular heart rate detected ({hr} bpm). Try to relax.'
        else:
             recommendation = 'We detected an unusual reading. Monitor your symptoms and rest.'

    # 5. GUARDAR os dados na base de dados (novo código SQLAlchemy)
    try:
        nova_leitura = HealthReading(
            heart_rate=hr,
            blood_oxygen=o2,
            status=status,
            recommendation=recommendation
        )
        db.session.add(nova_leitura)
        db.session.commit()
    except Exception as e:
        print(f"Erro ao guardar na DB: {e}")
        db.session.rollback()

    # 6. LER o histórico da base de dados (novo código SQLAlchemy)
    readings_from_db = HealthReading.query.order_by(HealthReading.timestamp.desc()).limit(20).all()

    # Preparar dados para o gráfico
    history = {
        'labels': [to_str(r.timestamp.strftime('%Y-%m-%d %H:%M:%S')).split(' ')[1] for r in reversed(readings_from_db)],
        'heart_rates': [r.heart_rate for r in reversed(readings_from_db)],
        'blood_oxygens': [r.blood_oxygen for r in reversed(readings_from_db)]
    }

    # 7. Enviar todos os dados para a página web
    health_data = {
        'heart_rate': hr,
        'blood_oxygen': o2,
        'status': status,
        'recommendation': recommendation
    }
    
    return render_template('index.html', data=health_data, history=history)

# --- Endpoint da API (Adicionado) ---
@app.route('/api/predict', methods=['POST'])
def api_predict():
    # (Este é o código para a tua API, que discutimos antes)
    # (Não está totalmente implementado aqui, mas a estrutura está pronta)
    pass

# --- Código para iniciar a aplicação ---
if __name__ == '__main__':
    init_db() # Cria as tabelas quando executas localmente
    app.run(debug=True)
else:
    # Isto é o que o Render vai usar para criar as tabelas na primeira vez
    init_db()