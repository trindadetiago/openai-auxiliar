from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")
client = OpenAI()

models = ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_embedding_dataframe(df, column, model="text-embedding-3-large"):
    embeddings_kept = {}
    unique_values = df[column].unique()
    embeddings_kept = {value: get_embedding(value, model=model) for value in unique_values}

    df[column + "_embedding"] = df[column].apply(lambda x: embeddings_kept[x] if x in embeddings_kept else None)
    return df

from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import numpy as np

def get_similarity(text1, text2, model="text-embedding-3-large"):
    embedding1 = get_embedding(text1, model=model)
    embedding2 = get_embedding(text2, model=model)
    return cosine_similarity([embedding1], [embedding2])[0][0]

def get_similarity_dataframe(df, column1, text, model="text-embedding-3-large"):
    df["similarity"] = df.apply(lambda x: get_similarity(x[column1], text, model=model), axis=1)
    return df

def get_n_most_similar_from_0(df, text, column, n=5, model="text-embedding-3-large"):
    df = get_similarity_dataframe(df, column, text, model=model)
    if n == 0:
        return df.sort_values("similarity", ascending=False)
    return df.nlargest(n, "similarity")
    
def get_n_most_similar_from_df(df, text, column, n=5, model="text-embedding-3-large"):
    text_embedding = get_embedding(text, model=model)
    try:
        df["embedding"] = df[column].apply(literal_eval).apply(np.array)
    except:
        df["embedding"] = df[column]
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity([text_embedding], [x])[0][0])
    if n == 0:
        return df.sort_values("similarity", ascending=False)
    return df.nlargest(n, "similarity")