import streamlit as st
from embeddings import get_embedding, get_embedding_dataframe, get_n_most_similar_from_df
from cluestering import cluster, optimal_cluster
import pandas as pd


st.set_page_config(page_title="Embeddings Creator", page_icon="🦁", layout="wide")

models = ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']

st.title("Embeddings")

st.write('---')

tabs = st.tabs(["Text to Embedding", "Dataframe to Embeddings", "Similarity (Upload Embeddings)", "Clustering (Upload Embeddings)"])

if "embeddings_2" not in st.session_state:
    st.session_state["embeddings_2"] = None

with tabs[0]:
    input_text = st.text_input("Enter some text:")

    model = st.selectbox("Selecione um modelo", models, key="model1")

    if st.button("Process"):
        embedding = get_embedding(input_text, model=model)
        
        st.write("Embedding dimensions: ", len(embedding))
        df = pd.DataFrame({"text": [input_text], "embedding": [embedding]})
        st.dataframe(df, use_container_width=True)

with tabs[1]:
    uploaded_file = st.file_uploader("Entre um arquivo CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.write(df.shape)

        selected_column = st.selectbox("Selecione uma coluna", df.columns)

        st.write("Valores únicos: ", df[selected_column].nunique())

        model = st.selectbox("Selecione um modelo", models, key="model2")

        n_lines = st.slider("Número de linhas para comparar", min_value=1, max_value=df.shape[0], value=5)

        df = df.head(n_lines)

        if st.button("Criar embeddings"):
            df = get_embedding_dataframe(df, selected_column, model=model)

            st.session_state.embeddings_2 = df

        if st.session_state.embeddings_2 is not None:
            df = st.session_state.embeddings_2

            st.dataframe(df, use_container_width=True)

            text_to_compare = st.text_input("Digite um texto para comparar similaridades")

            if not text_to_compare:
                st.write("Por favor, insira um texto para comparar.")

            else:
                n = st.number_input("Número de resultados", min_value=0, value=5, key="n1")

                if st.button("Calcular similaridades"):
                    most_similar = get_n_most_similar_from_df(df, text_to_compare, selected_column+"_embedding", n=n, model=model)
                    st.dataframe(most_similar, use_container_width=True, hide_index=True)



with tabs[2]:
    uploaded_file = st.file_uploader("Entre um arquivo CSV", type=["csv"], key="file2")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), hide_index=True)
        st.write(df.shape)

        selected_column = st.selectbox("Selecione a primeira coluna", df.columns)

        text_to_compare = st.text_input("Digite um texto para comparar similaridades", key="text2")

        if not text_to_compare:
            st.write("Please enter a text to compare.")
        else:
            model = st.selectbox("Selecione um modelo", models, key="model3")

            n = st.number_input("Número de resultados", min_value=0, value=5, key="n2")

            embeddings = df[selected_column].tolist()
            if isinstance(embeddings, list):
                if st.button("Calcular similaridades", key="button2"):
                    most_similar = get_n_most_similar_from_df(df, text_to_compare, selected_column, n=n, model=model)
                    st.dataframe(most_similar, use_container_width=True, hide_index=True)
            else:
                st.write("Embeddings inválidas.")


from ast import literal_eval
import numpy as np

if 'embeddings_3' not in st.session_state:
    st.session_state['embeddings_3'] = None
        
with tabs[3]:
    uploaded_file = st.file_uploader("Entre um arquivo CSV", type=["csv"], key="file3")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['embeddings_3'] = df

    if st.session_state['embeddings_3'] is not None:
        df = st.session_state['embeddings_3']
        st.dataframe(df.head(), hide_index=True)
        st.write(df.shape)

        selected_column = st.selectbox("Selecione a coluna com as embeddings", df.columns)

        automatic_clusters = st.checkbox("Clusterização automática", value=True, key="auto_clusters")
        
        if not automatic_clusters:
            n_clusters = st.number_input("Número de clusters", min_value=2, value=4, key="n_clusters")
        else:
            method = st.selectbox("Método de seleção de clusters", ["silhouette", "elbow"], key="method")
        
        valid = False
        try:
            df[selected_column].apply(literal_eval).apply(np.array)
            valid = True
        except:
            st.write("Embeddings inválidas.")

        if valid:
            if st.button("Clusterizar", key="button3"):
                with st.spinner("Aguarde..."):
                    if automatic_clusters:
                        df, centroids = optimal_cluster(df, selected_column, method=method)
                    else:
                        df, centroids = cluster(df, selected_column, n_clusters=n_clusters)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    unique_clusters = df['Cluster'].unique()

                    for c in unique_clusters:
                        st.write(f"Cluster {c}:")
                        cluster_df = df[df['Cluster'] == c]
                        st.dataframe(cluster_df, use_container_width=True)
                        st.write("Tamanho: ", len(cluster_df))
                        st.write(f"Valores mais próximos do centróide do cluster {c}: ", centroids[c])