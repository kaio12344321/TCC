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
    # Lógica para categorizar a estratégia com base nas características disponíveis
    # Exemplo: implemente a lógica com base em padrões específicos nos logs
    if row['Distance_to_Ball'] > mediana_alta:
        return 'ofensivo'
    elif row['Distance_to_Ball'] < mediana_baixa:
        return 'defensivo'
    else:
        return 'contra_ataque'


# Ordenar
data_filtered['Estrategia'] = data_filtered.apply(
    categorizar_estrategia, axis=1)
data_filtered['Estrategia'] = pd.Categorical(
    data_filtered['Estrategia'], categories=estrategias, ordered=True)

data_filtered.to_excel('Debug.xlsx', index=False)

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

# Criar o modelo Random Forest para estratégias
model_estrategia = RandomForestClassifier(n_estimators=100, random_state=42)
pipeline_estrategia = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', model_estrategia)])

# Treinar o modelo no conjunto de treinamento para estratégias
print("Treinando o modelo de estratégias no conjunto de treinamento...")
train_sizes, train_scores, test_scores = learning_curve(
    pipeline_estrategia, X_train, y_train, cv=5)
print("Treinamento do modelo de estratégias concluído.")


print(f'X_sizes: {train_sizes}')
print(f'train_sizes: {len(train_sizes)}')


# Plotar a curva de aprendizagem
plt.figure()
plt.title("Curva de Aprendizagem")
plt.xlabel("Número de Exemplos de Treinamento")
plt.ylabel("Acurácia")
plt.grid()

train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Acurácia de Treinamento")

plt.legend(loc="best")
plt.show()

# Avaliar acurácia no conjunto de teste para estratégias
print("Avaliando acurácia no conjunto de teste para estratégias...")
pipeline_estrategia.fit(X_train, y_train)
y_pred_test_estrategia = pipeline_estrategia.predict(X_test)
acuracia_teste_estrategia = accuracy_score(y_test, y_pred_test_estrategia)
print(
    f'Acurácia na previsão de estratégias no conjunto de teste: {acuracia_teste_estrategia * 100:.2f}%')

# Fazer previsões de estratégias para todas as linhas do arquivo CSV
print("Fazendo previsões de estratégias no conjunto completo...")
data_filtered['Predicao_Estrategia'] = pipeline_estrategia.predict(
    X_estrategia)
print("Previsões de estratégias concluídas.")

# Calcular acurácia das previsões de estratégias no conjunto completo
acuracia_treinamento_estrategia = accuracy_score(
    y_encoded_estrategia, data_filtered['Predicao_Estrategia'])
print(
    f'Acurácia das previsões de estratégias no conjunto completo: {acuracia_treinamento_estrategia * 100:.2f}%')

# Matriz de confusão
conf_matrix = confusion_matrix(
    y_encoded_estrategia, data_filtered['Predicao_Estrategia'])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predição")
plt.ylabel("Real")
plt.title("Matriz de Confusão Random Forest")
plt.show()

# Calcular a média da distância ao gol para cada time
mean_distance_blue = data_filtered.loc[data_filtered['Team']
                                       == 'azul', 'Distance_to_Ball'].mean()
mean_distance_yellow = data_filtered.loc[data_filtered['Team']
                                         == 'amarelo', 'Distance_to_Ball'].mean()

# Determinar se um time foi mais ofensivo, defensivo ou neutro
if mean_distance_blue and mean_distance_yellow:
    if mean_distance_blue < mean_distance_yellow:
        team_classification = 'O time azul foi mais ofensivo e o time amarelo foi mais defensivo.'
    elif mean_distance_blue > mean_distance_yellow:
        team_classification = 'O time amarelo foi mais ofensivo e o time azul foi mais defensivo.'
    else:
        team_classification = 'Ambos os times tiveram atuações equilibradas.'
elif mean_distance_blue:
    team_classification = 'O time azul foi mais ofensivo.'
elif mean_distance_yellow:
    team_classification = 'O time amarelo foi mais ofensivo.'
else:
    team_classification = 'Não há dados suficientes para classificar a atuação dos times.'

# Adicionar essa classificação ao final da análise
print("Análise da atuação dos times:")
print(team_classification)

# Excluir a linha 0 dos resultados
resultados_estrategia = data_filtered.loc[data_filtered.index != 0]

# Adicionar a coluna 'Robot' ao início do DataFrame de resultados
resultados_estrategia.insert(0, 'Robot', data['Robot'])

# Salvar os resultados de estratégias em um novo arquivo CSV
resultados_estrategia[['Robot', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y', 'Distance_to_Ball', 'Team',
                       'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                       'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                       'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE',
                       'Estrategia', 'Predicao_Estrategia']].to_csv('resultadosRF.csv', index=False)

# Visualizar os resultados
print(resultados_estrategia)

# Calcular o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo de execução total: {execution_time} segundos.")
