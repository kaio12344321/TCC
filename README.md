# Estudo de Eficiência e Movimentação de Robôs em Partidas de Futebol (RoboCup)

Este repositório contém os principais artefatos desenvolvidos durante o projeto de TCC intitulado **"Estudo de Eficiência e Movimentação de Robôs em Partidas de Futebol"**, com foco na aplicação de técnicas de **aprendizado de máquina supervisionado** para classificar estratégias adotadas por robôs nas partidas da **RoboCup (Small Size League)**.

---

## 📂 Organização do Repositório

```
📦 TCC/
├── 📁 codigos_modelos/         # Scripts dos modelos SVM, RF, MLP e LSTM
├── 📁 resultados/              # Imagens dos resultados (curvas, matrizes e relatórios)
├── 📁 dashboard/               # Capturas do painel interativo desenvolvido em Power BI
├── 📁 dados/                   # Base de dados utilizada nos experimentos (CSV)
└── 📄 README.md                # Documento atual com instruções e orientações
```

---

## 🧠 Modelos Desenvolvidos

Os seguintes algoritmos de aprendizado supervisionado foram implementados:

- `SVM` (Support Vector Machine)
- `Random Forest`
- `MLP` (Multi-Layer Perceptron)
- `LSTM` (Long Short-Term Memory)

Cada script inclui o pipeline completo:
1. Carregamento e pré-processamento da base de dados
2. Treinamento e avaliação dos modelos
3. Geração de métricas e visualizações

---

## 📊 Resultados e Visualizações

A pasta `resultados/` contém:

- 📈 **Curvas de aprendizado** de cada modelo
- 📉 **Matrizes de confusão** para visualização dos erros de classificação
- 📃 **Relatórios de classificação** (precision, recall, F1-score e acurácia)

Essas saídas foram utilizadas como base para análise comparativa entre os modelos.

---

## 📊 Dashboard Interativo

Foi desenvolvido um **painel interativo em Power BI** para facilitar a interpretação dos resultados dos modelos.

- O dashboard apresenta:
  - Estratégia por robô e por equipe
  - Posse de bola
  - Padrões de movimentação
  - Desempenho por partida
  - Outros indicadores visuais estratégicos

> ⚠️ Por limitações técnicas, o arquivo `.pbix` não está hospedado diretamente no repositório. No entanto, a pasta `dashboard/` inclui imagens ilustrativas da interface final.

---

## 🗃️ Base de Dados

A base **`Base de Dados TCC`** foi construída a partir de logs de partidas da RoboCup:

- Formato: `.csv`
- Dados contidos:
  - Posições dos robôs e da bola
  - Timestamp das jogadas
  - Ações em campo
  - Metadados das partidas

Ela foi **organizada e pré-processada manualmente** para alimentar os modelos e o dashboard de forma estruturada e compatível.

---

## ⚙️ Requisitos e Execução

### ✅ Requisitos

- Python 3.10+
- Bibliotecas:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `tensorflow` (para MLP e LSTM)

> Você pode instalar os requisitos com:
```bash
pip install -r requirements.txt
```

### ▶️ Executando os scripts

Cada script pode ser executado individualmente. Exemplo:
```bash
python modelo_mlp.py
```

Os arquivos de saída (predições, gráficos, relatórios) serão salvos automaticamente na pasta `resultados/`.

---

## 📌 Objetivo do Projeto

- Aplicar técnicas de **Machine Learning** para identificar **estratégias de jogo** em partidas simuladas da RoboCup.
- Criar um **painel analítico** que auxilie na **análise tática** e na **tomada de decisões estratégicas** por parte das equipes participantes.

---

## 📄 Relatório Acadêmico

Este repositório é parte integrante do Trabalho de Conclusão de Curso submetido ao **Centro Universitário FEI**, curso de **Ciência da Computação**, por:

- Kaio Henrique da Silva Souza  
- Murilo Zoia Jacomino  
- Nicolas Moretti Trevizam  
- Guilherme Brigagão Cabelo  

Orientador: Prof. Dr. Danilo Hernani Perico

---

## 📬 Contato

Para dúvidas, sugestões ou colaborações, entre em contato por e-mail:  
📧 **kaio3600@gmail.com**