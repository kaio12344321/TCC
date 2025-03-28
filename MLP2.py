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
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

# Iniciar o temporizador
start_time = time.time()

# Carregar dados, pulando a primeira linha
data = pd.read_csv('dados_robos_e_bola.csv', header=None, names=['Frame', 'Robot', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y',
                                                                 'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                                                                 'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                                                                 'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE'], skiprows=1)

# Converter colunas para tipos numéricos, ignorando possíveis erros
numeric_columns = ['Frame', 'X_Robot',
                   'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y']
data[numeric_columns] = data[numeric_columns].apply(
    pd.to_numeric, errors='coerce')

# Filtrar linhas onde 'Ball_X' ou 'Ball_Y' são 999999999.0
data_filtered = data[(data['Ball_X'] != 999999999.0) &
                     (data['Ball_Y'] != 999999999.0)].copy()

# Calcular características adicionais
data_filtered['Distance_to_Ball'] = ((data_filtered['X_Robot'] - data_filtered['Ball_X'])**2 + (
    data_filtered['Y_Robot'] - data_filtered['Ball_Y'])**2)**0.5

# Extrair informações sobre o time do robô (azul ou amarelo)
data_filtered['Team'] = data_filtered['Robot'].apply(
    lambda x: 'azul' if ('Robo B' in str(x)) else 'amarelo')

# Excluir as colunas 'Frame' e 'Robot'
data_filtered = data_filtered.drop(['Frame', 'Robot'], axis=1)

# Preencher valores NaN usando SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Codificar variáveis categóricas usando OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        # Padronização dos dados na mesma escala de valores
        ('num', StandardScaler(), numeric_columns[1:]),
        ('cat', OneHotEncoder(), ['Team'])
    ]
)

# Calcular medianas alta e baixa
mediana_alta = data_filtered['Distance_to_Ball'].median(
) + data_filtered['Distance_to_Ball'].std()
mediana_baixa = data_filtered['Distance_to_Ball'].median(
) - data_filtered['Distance_to_Ball'].std()

# Aplicar a função de categorização


def categorizar_posicao(row):
    if row['Distance_to_Ball'] > mediana_alta:
        return 'ofensivo'
    elif row['Distance_to_Ball'] < mediana_baixa:
        return 'passivo'
    else:
        return 'neutro'


data_filtered['Posicao'] = data_filtered.apply(categorizar_posicao, axis=1)

# Convertendo a coluna 'Posicao' para categórica (colocando índices) não são mais tratadas como texto
data_filtered['Posicao'] = pd.Categorical(data_filtered['Posicao'], categories=[
                                          'ofensivo', 'passivo', 'neutro'], ordered=True)

# Definir estratégias
estrategias = ['defensivo', 'ofensivo', 'contra_ataque']

# Criar variável alvo para estratégias


def categorizar_estrategia(row):
    if row['Distance_to_Ball'] > mediana_alta:
        return 'ofensivo'
    elif row['Distance_to_Ball'] < mediana_baixa:
        return 'defensivo'
    else:
        return 'contra_ataque'


# Convertendo a coluna 'Estrategia' para categórica (colocando índices) não são mais tratadas como texto
data_filtered['Estrategia'] = data_filtered.apply(
    categorizar_estrategia, axis=1)
data_filtered['Estrategia'] = pd.Categorical(
    data_filtered['Estrategia'], categories=estrategias, ordered=True)

# Preparar os dados para a nova tarefa de classificação de estratégias
X_estrategia = data_filtered[['X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y', 'Team',
                              'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                              'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                              'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE']]

# Transforma em índices a coluna estratégia --> Transforma-se a variável categórica y_estrategia (estratégia do robô) em valores numéricos
y_estrategia = data_filtered['Estrategia']
le_estrategia = LabelEncoder()
y_encoded_estrategia = le_estrategia.fit_transform(y_estrategia)

# Dividir os dados em conjuntos de treinamento e teste (80% para treinamento, 20% para teste)
X_train, X_test, y_train, y_test = train_test_split(
    X_estrategia, y_encoded_estrategia, test_size=0.2, random_state=42)

print(X_train)

# Função para criar a MLP

# Ajustar os dados com o pré-processador
X_train_transformed = preprocessor.fit_transform(X_train)
print(len(X_train_transformed[0]))

X_test_transformed = preprocessor.transform(X_test)

# Criar o modelo MLP
model = Sequential()
model.add(Dense(input_dim=7, units=5, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train_transformed, y_train, epochs=50, batch_size=32)
