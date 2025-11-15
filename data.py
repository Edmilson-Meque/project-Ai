import pandas as pd
import numpy as np

def simulate_health_data(records=1000):
    """
    Simula um conjunto de dados de saúde para treino,
    agora com uma coluna 'is_anomaly' para avaliação.
    """
    data = {
        'timestamp': pd.date_range(start='2023-10-01', periods=records, freq='T'),
        'heart_rate': np.random.randint(60, 100, records),
        'blood_oxygen': np.random.randint(90, 100, records),
        'is_anomaly': np.zeros(records, dtype=int) # Coluna de gabarito
    }
    
    # Adicionar algumas anomalias para o modelo aprender
    # 10% dos registos serão anomalias
    anomaly_indices = np.random.choice(records, int(records * 0.1), replace=False)
    
    for i in anomaly_indices:
        data['is_anomaly'][i] = 1 # Marcar como anomalia
        
        if np.random.rand() > 0.5:
            # Anomalia de Frequência Cardíaca
            data['heart_rate'][i] = np.random.choice([np.random.randint(40, 55), np.random.randint(105, 130)])
        else:
            # Anomalia de Oxigénio
            data['blood_oxygen'][i] = np.random.randint(85, 90)

    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    df = simulate_health_data(10)
    print("--- Amostra de Dados Simulados (com Gabarito) ---")
    print(df)