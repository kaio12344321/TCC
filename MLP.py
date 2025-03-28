import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, learning_curve
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from matriz import create_matrix  # Importa a função do outro arquivo

# Iniciar o temporizador
start_time = time.time()

# Carregar dados, pulando a primeira linha
data = pd.read_csv('dados_robos_e_bola.csv', header=None, names=['Frame', 'Robot', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y',
                                                                 'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                                                                 'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                                                                 'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE'], skiprows=1)

# Converter colunas para tipos numéricos, ignorando possíveis erros
numeric_columns = ['Frame', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Filtrar linhas inválidas
data_filtered = data[(data['Ball_X'] != 999999999.0) & (data['Ball_Y'] != 999999999.0)].copy()

# Calcular distância da bola
data_filtered['Distance_to_Ball'] = ((data_filtered['X_Robot'] - data_filtered['Ball_X'])**2 +
                                     (data_filtered['Y_Robot'] - data_filtered['Ball_Y'])**2)**0.5

# Adicionar coluna de time
data_filtered['Team'] = data_filtered['Robot'].apply(lambda x: 'azul' if ('Robo B' in str(x)) else 'amarelo')

# Remover colunas desnecessárias
data_filtered = data_filtered.drop(['Frame', 'Robot'], axis=1)

# Imputação e pré-processamento
imputer = SimpleImputer(strategy='mean')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns[1:]),
        ('cat', OneHotEncoder(), ['Team'])
    ]
)

# Categorização
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

# Features e target
X_estrategia = data_filtered[['X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y', 'Team',
                              'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                              'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                              'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE']]

y_estrategia = data_filtered['Estrategia']
le_estrategia = LabelEncoder()
y_encoded_estrategia = le_estrategia.fit_transform(y_estrategia)

# Divisão de dados
X_train, X_test, y_train, y_test = train_test_split(X_estrategia, y_encoded_estrategia, test_size=0.2, random_state=42)

# Gera configurações de camadas ocultas
matrix_config = create_matrix(10)

# Loop por cada configuração
for row in matrix_config:
    print(f"\nTreinando MLP com camadas ocultas: {row}")

    # Pré-processamento
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Modelo
    model = Sequential()
    model.add(Dense(input_dim=X_train_transformed.shape[1], units=row[0], activation='relu'))
    model.add(Dense(units=row[1], activation='relu'))
    model.add(Dense(units=row[2], activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Treinamento
    history = model.fit(X_train_transformed, y_train, epochs=50, batch_size=32, verbose=0)

    # Avaliação
    loss, accuracy = model.evaluate(X_test_transformed, y_test, verbose=0)
    print(f"Acurácia: {accuracy:.4f} | Loss: {loss:.4f}")

# Exibir tempo total
print(f"\nTempo total de execução: {time.time() - start_time:.2f} segundos")
