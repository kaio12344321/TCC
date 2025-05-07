
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

start_time = time.time()
print("Início do carregamento e pré-processamento...")

data = pd.read_csv('dados_robos_e_bola.csv', header=None, skiprows=1)
data.columns = ['Frame', 'Robot', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y',
                'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE',
                'Referee_TeamYellowC_YELLOW', 'Referee_TeamYellowC_BLUE',
                'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE']

data = data.sort_values(by=['Robot', 'Frame']).reset_index(drop=True)
data = data[(data['Ball_X'] != 999999999.0) & (data['Ball_Y'] != 999999999.0)].copy()
data['Distance_to_Ball'] = ((data['X_Robot'] - data['Ball_X'])**2 + (data['Y_Robot'] - data['Ball_Y'])**2)**0.5

mediana = data['Distance_to_Ball'].median()
desvio = data['Distance_to_Ball'].std()
lim_sup = mediana + desvio
lim_inf = mediana - desvio

def classificar_estrategia(dist):
    if dist > lim_sup:
        return 'ofensivo'
    elif dist < lim_inf:
        return 'defensivo'
    else:
        return 'contra_ataque'

data['Estrategia'] = data['Distance_to_Ball'].apply(classificar_estrategia)
data['Team'] = data['Robot'].apply(lambda x: 'azul' if 'Robo B' in str(x) else 'amarelo')
data['Estrategia'] = pd.Categorical(data['Estrategia'], categories=['defensivo', 'ofensivo', 'contra_ataque'], ordered=True)
le_estrategia = LabelEncoder()
data['Estrategia_Encoded'] = le_estrategia.fit_transform(data['Estrategia'])
data = data.drop(columns=['Frame', 'Robot'])

feature_cols = ['X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y', 'Team',
                'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE',
                'Referee_TeamYellowC_YELLOW', 'Referee_TeamYellowC_BLUE',
                'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE']
target_col = 'Estrategia_Encoded'

seq_len = 5
robots = pd.read_csv('dados_robos_e_bola.csv', header=None, skiprows=1)[1]
robots_filtered = robots.loc[data.index].reset_index(drop=True)
X = data[feature_cols].reset_index(drop=True)
y = data[target_col].reset_index(drop=True)

X_seq, y_seq = [], []
for robot in robots_filtered.unique():
    mask = robots_filtered == robot
    X_robot = X[mask].reset_index(drop=True)
    y_robot = y[mask].reset_index(drop=True)
    for i in range(len(X_robot) - seq_len + 1):
        X_seq.append(X_robot.iloc[i:i+seq_len].to_numpy())
        y_seq.append(y_robot.iloc[i+seq_len-1])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)

num_cols = ['X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y']
cat_cols = ['Team']
X_reshaped = X.reshape(-1, X.shape[2])
df_seq = pd.DataFrame(X_reshaped, columns=num_cols + cat_cols + feature_cols[6:])

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(), cat_cols)
])

X_transformed = preprocessor.fit_transform(df_seq)
X_transformed = X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed
X_final = X_transformed.reshape(X.shape[0], seq_len, -1)

X_train, X_test = X_final[:len(X_train)], X_final[len(X_train):]

def build_lstm_model(hidden_layers, input_shape):
    model = Sequential()
    model.add(LSTM(hidden_layers[0], input_shape=input_shape, return_sequences=True))
    model.add(LSTM(hidden_layers[1], return_sequences=True))
    model.add(LSTM(hidden_layers[2]))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
camadas_config = [[3, 5, 10], [5, 5, 5], [10, 10, 10], [15, 15, 15]]
resultados = []

for layers in camadas_config:
    print(f"\nTreinando arquitetura LSTM com camadas: {layers}")
    model = build_lstm_model(layers, input_shape=(seq_len, X_final.shape[2]))
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

    # Curva de aprendizado
    plt.figure()
    plt.plot(history.history['accuracy'], label='Acurácia Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
    plt.title(f"Curva de Aprendizado - LSTM {layers}")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Avaliação
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Acurácia: {acc:.4f} | Loss: {loss:.4f}")

    # Classificação
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=le_estrategia.classes_, output_dict=True)).transpose()
    report_df.to_csv(f"classification_report_lstm_{layers[0]}_{layers[1]}_{layers[2]}.csv")

    # Matriz de confusão
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
                xticklabels=le_estrategia.classes_, yticklabels=le_estrategia.classes_)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão - LSTM {layers}')
    plt.tight_layout()
    plt.show()

    # Análise por time
    y_pred_all = np.argmax(model.predict(X_final), axis=1)
    data_copy = data.copy().iloc[seq_len-1:].reset_index(drop=True)
    data_copy['Predicao_Estrategia'] = y_pred_all
    mean_azul = data_copy[data_copy['Team'] == 'azul']['Distance_to_Ball'].mean()
    mean_amarelo = data_copy[data_copy['Team'] == 'amarelo']['Distance_to_Ball'].mean()

    print("\nAnálise dos times:")
    if mean_azul < mean_amarelo:
        print("Time azul mais ofensivo.")
    elif mean_amarelo < mean_azul:
        print("Time amarelo mais ofensivo.")
    else:
        print("Atuação equilibrada.")

    resultados.append({'Camada 1': layers[0], 'Camada 2': layers[1], 'Camada 3': layers[2],
                       'Acurácia': acc, 'Loss': loss})

df_resultados = pd.DataFrame(resultados)
df_resultados.to_excel('resultados_lstm.xlsx', index=False)

melhor = df_resultados.loc[df_resultados['Acurácia'].idxmax()]
print(f"\nMelhor arquitetura:")
print(melhor)

print(f"\nTempo total: {time.time() - start_time:.2f} segundos")
print("Fim do script.")
