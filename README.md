# Projeto 2: Detecção de Anomalias em Tempo Real com Autoencoder (LSTM)

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blueviolet?logo=scikit-learn)

## 1. Visão Geral do Projeto

Este é um projeto **end-to-end** de **Aprendizagem Não Supervisionada** focado na detecção de anomalias em dados de sensores industriais. Diferente da manutenção preditiva (que prevê *quando* uma falha conhecida ocorrerá), este projeto foca em detectar *comportamentos estranhos e desconhecidos* no momento em que eles acontecem.

Utilizando o dataset **SKAB (Skoltech Anomaly Benchmark)**, um modelo Autoencoder baseado em LSTM é treinado para "aprender" o comportamento normal de uma válvula industrial. O resultado final é um **Dashboard em tempo real (Streamlit)** que monitora um fluxo simulado de dados e dispara alertas visuais imediatos quando o comportamento do sistema se desvia da "normalidade".

## 2. O Problema de Negócio: Detectando o "Desconhecido"

Em sistemas industriais complexos, existem inúmeras formas de falha. Enquanto a "Manutenção Preditiva" (como no Projeto 1) é excelente para prever falhas *conhecidas* (ex: desgaste de rolamento), ela falha em detectar eventos raros, súbitos ou nunca antes vistos.

O desafio é: **Como podemos detectar uma falha se não temos dados de exemplo dessa falha para treinar o modelo?**

A solução é a **Detecção de Anomalias (Aprendizagem Não Supervisionada)**. Em vez de treinar o modelo para reconhecer falhas, nós o treinamos para ser um "especialista" em reconhecer a operação *normal*. Qualquer dado que o especialista não reconheça é, por definição, uma anomalia.

## 3. A Solução de Machine Learning: O Autoencoder

A arquitetura do **Autoencoder** é perfeitamente adequada para isso. O modelo funciona como um "funil":

1.  **Encoder (Codificador):** Comprime os dados de entrada (ex: 20 ciclos com 8 sensores) em uma representação compacta (um "gargalo" ou *bottleneck*). Isso força o modelo a aprender a essência dos dados, descartando o ruído.
2.  **Decoder (Decodificador):** Tenta reconstruir perfeitamente os dados de entrada originais a partir dessa representação compacta.

**O Processo de Detecção:**

1.  **Treinamento:** O Autoencoder é treinado **exclusivamente com dados 100% normais**. Ele se torna excelente em comprimir e reconstruir a "assinatura" da operação normal, resultando em um **Erro de Reconstrução** muito baixo.
2.  **Definição do Limiar (Threshold):** Calculamos o erro médio ($\mu$) e o desvio padrão ($\sigma$) dos dados normais. Definimos um limiar estatístico (ex: $\mu + 3\sigma$). Qualquer erro *acima* deste limiar é considerado estatisticamente improvável de ser "normal".
3.  **Detecção:** Quando o modelo treinado recebe um dado novo:
    * **Dado Normal:** O modelo o reconstrói facilmente. O erro fica *abaixo* do limiar. -> **Status: OK.**
    * **Dado Anômalo:** O modelo (treinado só em normalidade) falha em reconstruí-lo. O erro *dispara* e ultrapassa o limiar. -> **Status: ANOMALIA DETECTADA.**

## 4. O Dashboard Interativo (Streamlit)

O produto final é um dashboard em `streamlit` que simula um cenário de monitoramento em tempo real. O dashboard:
* Carrega o modelo `autoencoder_model.h5` e o `anomaly_scaler.pkl`.
* Permite que o usuário insira o `threshold` (Limiar) calculado no notebook.
* Simula um streaming de dados (lendo o arquivo `1.csv`, que contém anomalias).
* Plota o Erro de Reconstrução em um gráfico de linha.
* Exibe um alerta visual (vermelho) e um status de "ANOMALIA DETECTADA" sempre que o erro ultrapassa o limiar.

*(Exemplo do Dashboard em ação)*
`![Dashboard de Detecção de Anomalias](dashboard_demo.png)`

## 5. Tech Stack (Tecnologias Utilizadas)

* **Dashboard e Visualização:** Streamlit
* **Deep Learning:** TensorFlow (Keras) (para o Autoencoder LSTM)
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (para `MinMaxScaler` e métricas)
* **Serialização:** Joblib

## 6. Estrutura do Projeto

```
[ projeto_deteccao_anomalia/ ]
|
|-- data/
|   |-- raw/          <-- (CSVs do SKAB, ex: '1.csv', '2.csv'...)
|   |-- processed/
|       |-- normal_data_train.csv  <-- (Gerado pelo notebook)
|
|-- dashboard/
|   |-- app.py                  <-- (Script do Streamlit)
|   |-- requirements_dash.txt
|
|-- models/
|   |-- autoencoder_model.h5    <-- (Modelo treinado)
|   |-- anomaly_scaler.pkl      <-- (Scaler treinado)
|
|-- notebooks/
|   |-- 01_EDA_e_Treinamento_Autoencoder.ipynb
|
|-- requirements.txt            <-- (Dependências do notebook)
|-- README.md
```

## 7. Como Usar

### Pré-requisitos
* Python 3.11+
* Ambiente virtual (recomendado)

### Instalação
1.  Clone o repositório e entre na pasta.
2.  Crie e ative um ambiente virtual.
3.  Instale as dependências de treinamento:
    ```bash
    pip install -r requirements.txt
    ```
4.  Instale as dependências do dashboard:
    ```bash
    pip install -r dashboard/requirements_dash.txt
    ```

### Passo 1: Treinar o Modelo
1.  Abra e execute o notebook `notebooks/01_EDA_e_Treinamento_Autoencoder.ipynb`.
2.  Execute todas as células (Célula 1, 2, 3 e 4).
3.  Ao final da **Célula 3**, anote o valor do limiar calculado:
    `Limiar (μ + 3σ): 0.XXXXXX`
    
Isso salvará os arquivos `autoencoder_model.h5` e `anomaly_scaler.pkl` na pasta `models/`.

### Passo 2: Executar o Dashboard
1.  No seu terminal, a partir da pasta raiz do projeto, execute o Streamlit:
    ```bash
    streamlit run dashboard/app.py
    ```
2.  O Streamlit abrirá no seu navegador.
3.  Na barra lateral esquerda, **cole o valor do Limiar (threshold)** que você anotou.
4.  Clique no botão **"Iniciar Simulação 🚀"**.

Observe o gráfico de erro em tempo real e os alertas de anomalia.

## 8. Resultados do Modelo

A avaliação do modelo (na Célula 4 do notebook) contra os dados completos (incluindo anomalias) produziu os seguintes resultados:

* **Precision (Anomalia): 0.98 (98%)**
    * **Interpretação:** Quando o modelo dispara um alarme de anomalia, ele está correto 98% das vezes. Isso é excelente, pois minimiza "alarmes falsos" (Falsos Positivos).

* **Recall (Anomalia): 0.56 (56%)**
    * **Interpretação:** O modelo conseguiu detectar 56% de todas as anomalias reais presentes nos dados.

* **O Trade-off (Precisão vs. Recall):**
    O Recall de 56% é um resultado direto da nossa escolha de um limiar estatístico rigoroso (3 sigmas), que prioriza a **alta precisão**. Em um cenário real, este limiar é um "botão de sintonia":
    * **Aumentar o Limiar:** Diminui os alarmes falsos (aumenta a precisão), mas pode perder falhas sutis (diminui o recall).
    * **Diminuir o Limiar:** Encontra mais falhas (aumenta o recall), mas ao custo de mais alarmes falsos (diminui a precisão).
