import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, learning_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # Adicionando importação do numpy

# Iniciar o temporizador
start_time = time.time()

# Carregar dados, pulando a primeira linha
print("Carregando dados...")
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

# Codificar variáveis categóricas usando LabelEncoder
label_encoders = {}
categorical_columns = ['Team']
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data_filtered[column] = label_encoders[column].fit_transform(
        data_filtered[column])

# Calcular medianas alta e baixa
mediana_alta = data_filtered['Distance_to_Ball'].median(
) + data_filtered['Distance_to_Ball'].std()
mediana_baixa = data_filtered['Distance_to_Ball'].median(
) - data_filtered['Distance_to_Ball'].std()

# Aplicar a função de categorização Estratégia
def categorizar_estrategia(row):
    # Lógica para categorizar a estratégia com base nas características disponíveis
    # Exemplo: implemente a lógica com base em padrões específicos nos logs
    if row['Distance_to_Ball'] > mediana_alta:
        return 'ofensivo'
    elif row['Distance_to_Ball'] < mediana_baixa:
        return 'defensivo'
    else:
        return 'contra_ataque'


data_filtered['Estrategia'] = data_filtered.apply(categorizar_estrategia, axis=1)

# Convertendo a coluna 'Estrategia' para categórica
data_filtered['Estrategia'] = pd.Categorical(data_filtered['Estrategia'], categories=['defensivo', 'ofensivo', 'contra_ataque'], ordered=True)

# Dividir os dados em conjuntos de treinamento e teste (80% para treinamento, 20% para teste)
X_estrategia = data_filtered[['X_Robot', 'Y_Robot',
                              'Orientation', 'Ball_X', 'Ball_Y', 'Team']]
y_estrategia = data_filtered['Estrategia']
X_train, X_test, y_train, y_test = train_test_split(
    X_estrategia, y_estrategia, test_size=0.2, random_state=42)

# Criar o modelo SVM para estratégias
model_estrategia = SVC(kernel='rbf', random_state=42)

# Criar o pipeline
pipeline_estrategia = Pipeline(steps=[
    ('imputer', imputer),
    ('scaler', StandardScaler()),
    ('classifier', model_estrategia)
])


print("Treinando o modelo SVM para estratégias...")
# Treinar o modelo no conjunto de treinamento para estratégias
pipeline_estrategia.fit(X_train, y_train)

# Calcular curva de aprendizagem
print("Calculando curva de aprendizagem...")
train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipeline_estrategia,
    X=X_train,
    y=y_train,
    cv=5,
    n_jobs=-1
)

# Curva de aprendizado com treino e validação 
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Acurácia Treino')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Acurácia Validação')
plt.title("Curva de Aprendizado (SVM)")
plt.xlabel("Número de Exemplos de Treinamento")
plt.ylabel("Acurácia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Avaliar acurácia no conjunto de teste para estratégias
y_pred_test_estrategia = pipeline_estrategia.predict(X_test)
acuracia_teste_estrategia = accuracy_score(y_test, y_pred_test_estrategia)
print(f'Acurácia na previsão de estratégias no conjunto de teste: {acuracia_teste_estrategia * 100:.2f}%')

# Relatório de classificação
print("\nGerando relatório de classificação...")
print(classification_report(y_test, y_pred_test_estrategia, target_names=y_estrategia.cat.categories))

# Fazer previsões de estratégias para todas as linhas do arquivo CSV
print("Fazendo previsões de estratégias no conjunto completo...")
data_filtered['Predicao_Estrategia'] = pipeline_estrategia.predict(X_estrategia)


# Matriz de confusão no conjunto de teste
conf_matrix = confusion_matrix(y_test, y_pred_test_estrategia)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=y_estrategia.cat.categories,
            yticklabels=y_estrategia.cat.categories)
plt.xlabel("Predição")
plt.ylabel("Real")
plt.title("Matriz de Confusão (SVM)")
plt.show()

# Excluir a linha 0 dos resultados
resultados_estrategia = data_filtered.loc[data_filtered.index != 0]

# Adicionar a coluna 'Robot' ao início do DataFrame de resultados
resultados_estrategia.insert(0, 'Robot', data['Robot'])

# Salvar os resultados de estratégias em um novo arquivo CSV
resultados_estrategia[['Robot', 'X_Robot', 'Y_Robot', 'Orientation', 'Ball_X', 'Ball_Y', 'Distance_to_Ball', 'Team',
                       'Referee_Stage', 'Referee_TeamNames_YELLOW', 'Referee_TeamNames_BLUE',
                       'Referee_TeamScores_YELLOW', 'Referee_TeamScores_BLUE', 'Referee_TeamYellowC_YELLOW',
                       'Referee_TeamYellowC_BLUE', 'Referee_TeamRedC_YELLOW', 'Referee_TeamRedC_BLUE',
                       'Estrategia', 'Predicao_Estrategia']].to_csv('resultadosSVM.csv', index=False)

# Visualizar os resultados
print(resultados_estrategia)

# Calcular o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo de execução total: {execution_time} segundos.")
