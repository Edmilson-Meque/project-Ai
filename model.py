import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from data import simulate_health_data

def train_and_evaluate_model():
    """
    Treina, AVALIA e guarda o modelo de detecção de anomalias.
    """
    print("1. A simular dados de treino (5000 registos)...")
    df = simulate_health_data(records=5000)
    
    features = ['heart_rate', 'blood_oxygen']
    
    # A nossa coluna "gabarito" (0 = normal, 1 = anomalia)
    labels = df['is_anomaly'] 
    
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(df[features], labels, test_size=0.2, random_state=42)

    print("2. A treinar o modelo de detecção de anomalias (IsolationForest)...")
    # contamination=0.1 porque sabemos que simulámos 10% de anomalias
    model = IsolationForest(contamination=0.1, random_state=42)
    
    # O IsolationForest (não supervisionado) só precisa de X_train para treinar
    model.fit(X_train)

    # --- AVALIAÇÃO (O NOVO PASSO 4) ---
    print("\n3. A avaliar o modelo nos dados de teste...")
    
    # Obter previsões do modelo nos dados de teste
    # O modelo retorna 1 para normal (inlier) e -1 para anomalia (outlier)
    y_pred_test = model.predict(X_test)
    
    # Precisamos de "traduzir" a saída do modelo (-1, 1) para o nosso gabarito (1, 0)
    # Modelo: -1 = anomalia, 1 = normal
    # Gabarito: 1 = anomalia, 0 = normal
    y_pred_mapped = [1 if x == -1 else 0 for x in y_pred_test]

    print("\n--- Relatório de Classificação ---")
    print("IsolationForest (Mapeado):")
    print(classification_report(y_test, y_pred_mapped, target_names=['Normal (0)', 'Anomalia (1)']))
    
    print("--- Matriz de Confusão ---")
    print(confusion_matrix(y_test, y_pred_mapped))
    print("(Linhas = Real, Colunas = Previsto)")

    print("\n4. A guardar o modelo final (treinado em TODOS os dados) em 'model.joblib'...")
    # Para o modelo final, treinamos com todos os dados
    final_model = IsolationForest(contamination=0.1, random_state=42)
    final_model.fit(df[features])
    joblib.dump(final_model, 'model.joblib')
    
    print("--- Modelo treinado, AVALIADO e guardado com sucesso! ---")

if __name__ == '__main__':
    train_and_evaluate_model()