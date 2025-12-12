import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Análise de Depressão Estudantil",
    page_icon="brain",
    layout="wide"
)

@st.cache_resource
def carregar_modelo():
    with open('models/modelo_depressao.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('models/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return modelo, scaler, feature_names, model_info, label_encoders

@st.cache_data
def carregar_dados():
    df = pd.read_csv('data/student_depression_dataset.csv')
    return df

try:
    modelo, scaler, feature_names, model_info, label_encoders = carregar_modelo()
    df = carregar_dados()
    modelo_carregado = True
except:
    modelo_carregado = False
    st.error("Erro ao carregar o modelo. Execute os notebooks de treinamento primeiro.")

st.title("Sistema de Análise de Depressão em Estudantes")
st.markdown("Projeto de Machine Learning aplicado à Saúde Mental")

menu = st.sidebar.selectbox(
    "Navegação",
    ["Página Inicial", "Fazer Predição", "Análise dos Dados", "Sobre o Projeto"]
)

if menu == "Página Inicial":
    st.header("Bem-vindo ao Sistema de Análise")
    
    st.markdown("""
    Este sistema utiliza técnicas de Machine Learning para analisar fatores 
    relacionados à depressão em estudantes universitários utilizando .
    """)
    
    if modelo_carregado:
        col1, col2, col3, col4 = st.rows(4)
        
        with col1:
            st.metric("Modelo", model_info['modelo_nome'])
        with col2:
            st.metric("Acurácia", f"{model_info['acuracia']:.2%}")
        with col3:
            st.metric("F1-Score", f"{model_info['f1_score']:.2%}")
        with col4:
            st.metric("Amostras de Treino", model_info['n_samples_train'])
        
        st.subheader("Informações do Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Total de registros: {len(df)}")
            st.write(f"Número de features: {len(feature_names)}")
        
        with col2:
            st.write(f"Registros com depressão: {df[model_info['target_col']].sum()}")
            st.write(f"Registros sem depressão: {len(df) - df[model_info['target_col']].sum()}")

elif menu == "Fazer Predição":
    st.header("Realizar Nova Predição")
    
    if not modelo_carregado:
        st.warning("Modelo não carregado. Execute os notebooks de treinamento.")
    else:
        st.markdown("Preencha as informações abaixo para realizar a análise:")
        
        input_data = {}
        
        num_cols = 3
        cols = st.columns(num_cols)
        
        for idx, feature in enumerate(feature_names):
            col_idx = idx % num_cols
            
            with cols[col_idx]:
                if df[feature].dtype in ['int64', 'float64']:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=feature
                    )
                else:
                    unique_vals = df[feature].unique()
                    input_data[feature] = st.selectbox(
                        f"{feature}",
                        options=unique_vals,
                        key=feature
                    )
        
        if st.button("Realizar Análise", type="primary"):
            input_df = pd.DataFrame([input_data])
            
            for col in input_df.columns:
                if col in label_encoders:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            
            input_scaled = scaler.transform(input_df)
            
            predicao = modelo.predict(input_scaled)[0]
            probabilidade = modelo.predict_proba(input_scaled)[0]
            
            st.subheader("Resultado da Análise")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if predicao == 1:
                    st.error("Indicadores de risco detectados")
                    st.write("A análise sugere a presença de fatores de risco para depressão.")
                else:
                    st.success("Indicadores dentro da normalidade")
                    st.write("A análise não identificou fatores significativos de risco.")
            
            with col2:
                st.metric(
                    "Probabilidade de Risco",
                    f"{probabilidade[1]*100:.1f}%"
                )
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probabilidade[1]*100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if probabilidade[1] > 0.5 else "lightgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("Este sistema é uma ferramenta de apoio e não substitui avaliação profissional.")

elif menu == "Análise dos Dados":
    st.header("Análise Exploratória dos Dados")
    
    if not modelo_carregado:
        st.warning("Dados não carregados.")
    else:
        tab1, tab2, tab3 = st.tabs(["Visão Geral", "Distribuições", "Correlações"])
        
        with tab1:
            st.subheader("Visão Geral do Dataset")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Primeiras linhas do dataset:")
                st.dataframe(df.head(10))
            
            with col2:
                st.write("Estatísticas descritivas:")
                st.dataframe(df.describe())
            
            st.subheader("Distribuição da Variável Target")
            
            target_counts = df[model_info['target_col']].value_counts()
            
            fig = px.pie(
                values=target_counts.values,
                names=['Sem Depressão', 'Com Depressão'],
                title='Proporção de casos'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Distribuições das Variáveis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if model_info['target_col'] in numeric_cols:
                numeric_cols.remove(model_info['target_col'])
            
            selected_col = st.selectbox("Selecione uma variável:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df, 
                    x=selected_col,
                    title=f'Distribuição de {selected_col}',
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df,
                    y=selected_col,
                    title=f'Boxplot de {selected_col}'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            fig = px.histogram(
                df,
                x=selected_col,
                color=model_info['target_col'],
                title=f'Distribuição de {selected_col} por classe',
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Matriz de Correlação")
            
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title='Correlação entre Variáveis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if model_info['target_col'] in numeric_df.columns:
                st.subheader("Correlação com a Variável Target")
                
                target_corr = corr_matrix[model_info['target_col']].drop(model_info['target_col']).sort_values(ascending=False)
                
                fig = px.bar(
                    x=target_corr.values,
                    y=target_corr.index,
                    orientation='h',
                    title=f'Correlação com {model_info["target_col"]}',
                    labels={'x': 'Correlação', 'y': 'Variável'}
                )
                st.plotly_chart(fig, use_container_width=True)

else:
    st.header("Sobre o Projeto")
    
    st.markdown("""
    ### Objetivo
    
    Este projeto tem como objetivo identificar fatores de risco para depressão 
    em estudantes universitários utilizando técnicas de Machine Learning.    
       
    ### Tecnologias Utilizadas
    
    Python 3.x
    Scikit-learn para Machine Learning
    Pandas e NumPy para manipulação de dados
    Streamlit para interface web
    Plotly para visualizações interativas
    
    ### Dataset
    
    [Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset) disponível no Kaggle, contendo informações sobre
    fatores acadêmicos, sociais e comportamentais de estudantes.   
    
    ### Autor
                
    Apolo Nicolas Ferreira Tenório, aluno de Análise e Desenvolvimento de Sistemas da Faculdade Senac PE.
                
    """)
    
    if modelo_carregado:
        st.subheader("Informações Técnicas do Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Algoritmo: {model_info['modelo_nome']}")
            st.write(f"Acurácia: {model_info['acuracia']:.4f}")
            st.write(f"Precisão: {model_info['precisao']:.4f}")
        
        with col2:
            st.write(f"Recall: {model_info['recall']:.4f}")
            st.write(f"F1-Score: {model_info['f1_score']:.4f}")
            st.write(f"Features utilizadas: {model_info['n_features']}")