import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns




# Iniciar o temporizador
start_time = time.time()
print("Início do carregamento e pré-processamento...")

# Carregar dados
data = pd.read_csv('dados_robos_e_bola.csv', header=None, names=['Frame', 'Robot', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y',
                                                                 'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                                                                 'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                                                                 'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE'], skiprows=1)

numeric_columns = ['Frame', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
data_filtered = data[(data['Ball_X'] != 999999999.0) & (data['Ball_Y'] != 999999999.0)].copy()
data_filtered['Distance_to_Ball'] = ((data_filtered['X_Robot'] - data_filtered['Ball_X'])**2 +
                                     (data_filtered['Y_Robot'] - data_filtered['Ball_Y'])**2)**0.5
data_filtered['Team'] = data_filtered['Robot'].apply(lambda x: 'azul' if ('Robo B' in str(x)) else 'amarelo')
data_filtered = data_filtered.drop(['Frame', 'Robot'], axis=1)

mediana_alta = data_filtered['Distance_to_Ball'].median() + data_filtered['Distance_to_Ball'].std()
mediana_baixa = data_filtered['Distance_to_Ball'].median() - data_filtered['Distance_to_Ball'].std()

def categorizar_estrategia(row):
    if row['Distance_to_Ball'] > mediana_alta:
        return 'ofensivo'
    elif row['Distance_to_Ball'] < mediana_baixa:
        return 'defensivo'
    else:
        return 'contra_ataque'

data_filtered['Estrategia'] = data_filtered.apply(categorizar_estrategia, axis=1)
data_filtered['Estrategia'] = pd.Categorical(data_filtered['Estrategia'], categories=['defensivo', 'ofensivo', 'contra_ataque'], ordered=True)

X_estrategia = data_filtered[['X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y', 'Team',
                              'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                              'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                              'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE']]
y_estrategia = data_filtered['Estrategia']
le_estrategia = LabelEncoder()
y_encoded_estrategia = le_estrategia.fit_transform(y_estrategia)

X_train, X_test, y_train, y_test = train_test_split(X_estrategia, y_encoded_estrategia, test_size=0.2, random_state=42, stratify=y_encoded_estrategia)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y']),
        ('cat', OneHotEncoder(), ['Team'])
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

matrix_config = [[3, 5, 10], [5, 5, 5], [10, 10, 10], [15, 15, 15]]

def build_model(hidden_layers):
    model = Sequential()
    model.add(Dense(units=hidden_layers[0], activation='relu', input_dim=X_train_transformed.shape[1]))
    model.add(Dense(units=hidden_layers[1], activation='relu'))
    model.add(Dense(units=hidden_layers[2], activation='relu'))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

resultados = []
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


for row in matrix_config:
    print(f"\nTreinando arquitetura MLP com camadas: {row}")

    
    model = build_model(row)
    history = model.fit(
    X_train_transformed, y_train,
    validation_data=(X_test_transformed, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
    )
    
    # Curva de aprendizado com base nas épocas
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    plt.figure()
    plt.plot(epochs_range, history.history['accuracy'], 'o-', label='Acurácia Treino')
    plt.plot(epochs_range, history.history['val_accuracy'], 'o-', label='Acurácia Validação')
    plt.title(f"Curva de Aprendizado (MLP) - Arquitetura {row}")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    
    print("Avaliação no conjunto de teste...")
    loss, accuracy = model.evaluate(X_test_transformed, y_test, verbose=0)
    print(f"Acurácia: {accuracy:.4f} | Loss: {loss:.4f}")
    
    print("Gerando relatório de classificação...")
    y_pred_prob = model.predict(X_test_transformed)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    report_text = classification_report(y_test, y_pred_classes, target_names=le_estrategia.classes_)
    print(report_text)
    
    # Salvar classification report em .csv
    report_dict = classification_report(y_test, y_pred_classes, target_names=le_estrategia.classes_, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    csv_filename = f"classification_report_{row[0]}_{row[1]}_{row[2]}.csv"
    df_report.to_csv(csv_filename)
    
    print("Gerando matriz de confusão...")
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le_estrategia.classes_, yticklabels=le_estrategia.classes_)
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão (MLP) - Arquitetura {row}")
    plt.tight_layout()
    plt.show()

    print("Realizando predições completas e análise dos times...")
    y_pred_all = np.argmax(model.predict(preprocessor.transform(X_estrategia)), axis=1)
    data_filtered_copy = data_filtered.copy()
    data_filtered_copy['Predicao_Estrategia'] = y_pred_all

    mean_distance_blue = data_filtered_copy[data_filtered_copy['Team'] == 'azul']['Distance_to_Ball'].mean()
    mean_distance_yellow = data_filtered_copy[data_filtered_copy['Team'] == 'amarelo']['Distance_to_Ball'].mean()

    print("\nAnálise da atuação dos times:")
    if mean_distance_blue and mean_distance_yellow:
        if mean_distance_blue < mean_distance_yellow:
            print('O time azul foi mais ofensivo e o time amarelo foi mais defensivo.')
        elif mean_distance_blue > mean_distance_yellow:
            print('O time amarelo foi mais ofensivo e o time azul foi mais defensivo.')
        else:
            print('Ambos os times tiveram atuações equilibradas.')
    elif mean_distance_blue:
        print('O time azul foi mais ofensivo.')
    elif mean_distance_yellow:
        print('O time amarelo foi mais ofensivo.')
    else:
        print('Não há dados suficientes para classificar a atuação dos times.')

    print("Resultados dessa arquitetura registrados.")
    resultados.append({
        'Camada 1': row[0],
        'Camada 2': row[1],
        'Camada 3': row[2],
        'Acurácia': accuracy,
        'Loss': loss
    })

print("Salvando os resultados finais...")
df_resultados = pd.DataFrame(resultados)
df_resultados.to_excel('resultados_mlp.xlsx', index=False)
melhor_resultado = df_resultados.loc[df_resultados['Acurácia'].idxmax()]
print("\nMelhor arquitetura encontrada:")
print(f"Camada 1: {melhor_resultado['Camada 1']}, "
      f"Camada 2: {melhor_resultado['Camada 2']}, "
      f"Camada 3: {melhor_resultado['Camada 3']} "
      f"=> Acurácia: {melhor_resultado['Acurácia']:.4f}, Loss: {melhor_resultado['Loss']:.4f}")
print(f"\nTempo total de execução: {time.time() - start_time:.2f} segundos")
print("Finalização do script.")
