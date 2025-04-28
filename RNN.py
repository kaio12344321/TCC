import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings("ignore", category=UserWarning)


# Fixar seeds
np.random.seed(42)
tf.random.set_seed(42)

# Início do temporizador
start_time = time.time()
print("Início do pré-processamento...")

# Carregar dados
data = pd.read_csv('dados_robos_e_bola.csv', header=None, names=['Frame', 'Robot', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y',
                                                                 'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                                                                 'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                                                                 'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE'], skiprows=1)

numeric_columns = ['Frame', 'X_Robot',
                   'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y']
data[numeric_columns] = data[numeric_columns].apply(
    pd.to_numeric, errors='coerce')
data_filtered = data[(data['Ball_X'] != 999999999.0) &
                     (data['Ball_Y'] != 999999999.0)].copy()
data_filtered['Distance_to_Ball'] = ((data_filtered['X_Robot'] - data_filtered['Ball_X'])**2 +
                                     (data_filtered['Y_Robot'] - data_filtered['Ball_Y'])**2)**0.5
data_filtered['Team'] = data_filtered['Robot'].apply(
    lambda x: 'azul' if ('Robo B' in str(x)) else 'amarelo')
data_filtered = data_filtered.drop(['Frame', 'Robot'], axis=1)

mediana_alta = data_filtered['Distance_to_Ball'].median(
) + data_filtered['Distance_to_Ball'].std()
mediana_baixa = data_filtered['Distance_to_Ball'].median(
) - data_filtered['Distance_to_Ball'].std()


def categorizar_estrategia(row):
    if row['Distance_to_Ball'] > mediana_alta:
        return 'ofensivo'
    elif row['Distance_to_Ball'] < mediana_baixa:
        return 'defensivo'
    else:
        return 'contra_ataque'


data_filtered['Estrategia'] = data_filtered.apply(
    categorizar_estrategia, axis=1)
data_filtered['Estrategia'] = pd.Categorical(data_filtered['Estrategia'], categories=[
                                             'defensivo', 'ofensivo', 'contra_ataque'], ordered=True)

X_estrategia = data_filtered[['X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y', 'Team',
                              'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                              'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                              'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE']]
y_estrategia = data_filtered['Estrategia']
le_estrategia = LabelEncoder()
y_encoded_estrategia = le_estrategia.fit_transform(y_estrategia)

# Pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [
         'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y']),
        ('cat', OneHotEncoder(), ['Team'])
    ]
)

X_transformed = preprocessor.fit_transform(X_estrategia)

# Função para criar sequências temporais


def criar_sequencias(X, y, seq_len=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


# Gerar sequências
X_seq, y_seq = criar_sequencias(X_transformed, y_encoded_estrategia, seq_len=5)
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

# Arquiteturas a testar: [LSTM1, LSTM2, Dense]
matrix_config = [[3, 5, 10], [5, 5, 5], [10, 10, 10], [15, 15, 15]]
resultados = []

early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

for config in matrix_config:
    print(f"\nTreinando LSTM com arquitetura: {config}")

    def build_lstm_model():
        model = Sequential()
        model.add(LSTM(config[0], activation='tanh', return_sequences=True, input_shape=(
            X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(config[1], activation='tanh'))
        model.add(Dense(config[2], activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    model = build_lstm_model()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

    # Avaliação
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Acurácia: {accuracy:.4f} | Loss: {loss:.4f}")

    # Relatório
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_classes,
          target_names=le_estrategia.classes_))

    # Salvar classification report
    report_dict = classification_report(
        y_test, y_pred_classes, target_names=le_estrategia.classes_, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    csv_filename = f"classification_report_lstm_{config[0]}_{config[1]}_{config[2]}.csv"
    df_report.to_csv(csv_filename)

    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_estrategia.classes_, yticklabels=le_estrategia.classes_)
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão - LSTM {config}")
    plt.tight_layout()
    plt.show()

    # Predição completa
    X_total_seq, _ = criar_sequencias(
        X_transformed, y_encoded_estrategia, seq_len=5)
    y_pred_all = np.argmax(model.predict(X_total_seq), axis=1)

    data_filtered_temp = data_filtered.iloc[5:].copy()
    data_filtered_temp['Predicao_Estrategia'] = y_pred_all

    mean_distance_blue = data_filtered_temp[data_filtered_temp['Team']
                                            == 'azul']['Distance_to_Ball'].mean()
    mean_distance_yellow = data_filtered_temp[data_filtered_temp['Team']
                                              == 'amarelo']['Distance_to_Ball'].mean()

    print("\nAnálise da atuação dos times:")
    if mean_distance_blue < mean_distance_yellow:
        print('O time azul foi mais ofensivo e o time amarelo mais defensivo.')
    elif mean_distance_blue > mean_distance_yellow:
        print('O time amarelo foi mais ofensivo e o time azul mais defensivo.')
    else:
        print('Atuação equilibrada entre os times.')

    resultados.append({
        'LSTM1': config[0],
        'LSTM2': config[1],
        'Dense': config[2],
        'Acurácia': accuracy,
        'Loss': loss
    })

# Resumo final
df_resultados = pd.DataFrame(resultados)
df_resultados.to_excel('resultados_lstm.xlsx', index=False)

melhor_resultado = df_resultados.loc[df_resultados['Acurácia'].idxmax()]
print("\nMelhor arquitetura:")
print(
    f"LSTM1: {melhor_resultado['LSTM1']}, LSTM2: {melhor_resultado['LSTM2']}, Dense: {melhor_resultado['Dense']}")
print(
    f"Acurácia: {melhor_resultado['Acurácia']:.4f}, Loss: {melhor_resultado['Loss']:.4f}")

print(f"\nTempo total de execução: {time.time() - start_time:.2f} segundos")
print("Finalização do script.")
