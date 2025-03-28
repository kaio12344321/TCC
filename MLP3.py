import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matriz import create_matrix  # Função que retorna a matriz de configurações

# Iniciar o temporizador
start_time = time.time()

# Carregar dados
data = pd.read_csv('dados_robos_e_bola.csv', header=None, names=['Frame', 'Robot', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y',
                                                                 'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                                                                 'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                                                                 'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE'], skiprows=1)

# Converter colunas numéricas
numeric_columns = ['Frame', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Filtrar dados inválidos
data_filtered = data[(data['Ball_X'] != 999999999.0) & (data['Ball_Y'] != 999999999.0)].copy()

# Calcular distância até a bola
data_filtered['Distance_to_Ball'] = ((data_filtered['X_Robot'] - data_filtered['Ball_X'])**2 +
                                     (data_filtered['Y_Robot'] - data_filtered['Ball_Y'])**2)**0.5

# Adicionar coluna de time
data_filtered['Team'] = data_filtered['Robot'].apply(lambda x: 'azul' if ('Robo B' in str(x)) else 'amarelo')

# Remover colunas desnecessárias
data_filtered = data_filtered.drop(['Frame', 'Robot'], axis=1)

# Pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y']),
        ('cat', OneHotEncoder(), ['Team'])
    ]
)

# Estratégia como variável alvo
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

# Separar features e target
X_estrategia = data_filtered[['X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y', 'Team',
                              'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                              'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                              'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE']]
y_estrategia = data_filtered['Estrategia']
le_estrategia = LabelEncoder()
y_encoded_estrategia = le_estrategia.fit_transform(y_estrategia)

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X_estrategia, y_encoded_estrategia, test_size=0.2, random_state=42)

# Obter configurações de rede
matrix_config = create_matrix(10)

# Lista para armazenar resultados
resultados = []

# Treinamento dos modelos
for row in matrix_config:
    print(f"\nTreinando MLP com camadas ocultas: {row}")

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model = Sequential()
    model.add(Dense(input_dim=X_train_transformed.shape[1], units=row[0], activation='relu'))
    model.add(Dense(units=row[1], activation='relu'))
    model.add(Dense(units=row[2], activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train_transformed, y_train, epochs=50, batch_size=32, verbose=0)

    loss, accuracy = model.evaluate(X_test_transformed, y_test, verbose=0)
    print(f"Acurácia: {accuracy:.4f} | Loss: {loss:.4f}")

    resultados.append({
        'Camada 1': row[0],
        'Camada 2': row[1],
        'Camada 3': row[2],
        'Acurácia': accuracy,
        'Loss': loss
    })

# Criar DataFrame e salvar em Excel
df_resultados = pd.DataFrame(resultados)
df_resultados.to_excel('resultados_mlp.xlsx', index=False)

# Exibir melhor resultado
melhor_resultado = df_resultados.loc[df_resultados['Acurácia'].idxmax()]
print("\nMelhor arquitetura encontrada:")
print(f"Camada 1: {melhor_resultado['Camada 1']}, "
      f"Camada 2: {melhor_resultado['Camada 2']}, "
      f"Camada 3: {melhor_resultado['Camada 3']} "
      f"=> Acurácia: {melhor_resultado['Acurácia']:.4f}, Loss: {melhor_resultado['Loss']:.4f}")

# Exibir tempo total de execução
print(f"\nTempo total de execução: {time.time() - start_time:.2f} segundos")
