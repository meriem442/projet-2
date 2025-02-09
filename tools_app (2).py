import numpy as np
import pandas as pd
import requests
import json
from st_click_detector import click_detector
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from io import BytesIO

# Fonction pour charger les données du fichier JSON
def load_movie_data(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)

# Chargement des données
movie_data = load_movie_data("movie_data_with_videos.json")


# Lien direct de téléchargement du fichier Google Drive
file_id = '1cEo3sSPwn4y3FKnEYEaPK5f2PWHTUtgX'
url = f'https://drive.google.com/uc?export=download&id={file_id}'

# Télécharger le fichier depuis Google Drive
response = requests.get(url)

# Vérifiez si la demande a réussi (code 200)
if response.status_code == 200:
    # Charger le fichier Parquet dans pandas à partir du contenu téléchargé
    data = pd.read_parquet(BytesIO(response.content))
    print(data.head())  # Affiche les premières lignes du dataframe
else:
    print(f"Erreur lors du téléchargement : {response.status_code}")

# Fonction pour rechercher un film par ID
def trouver_id(film_id: int, movie_data: list = movie_data):
    for film in movie_data:
        if film.get("id") == film_id:
            return film
    return None

# Fonction de gestion du clic sur un film

def get_clicked(movie_data: list, film_title: str, idx: int, categorie: str, annee: int = None, key_: bool = False):
    """
    Gère le clic sur un film pour une liste de films avec filtrage possible par année, par genre.
    """

    # Filtrage par année si une année est spécifiée
    if annee:
        movie_data = [film for film in movie_data if film.get("year") == annee]

    # Trouver le film par son index
    film = trouver_id(idx, movie_data)
    if not film:
        print(f"Aucun film trouvé pour l'ID : {idx}")
        return None, False
    
    film_id = film.get("id", None)
    film_title = film.get("title", "Titre inconnu")
    poster_path = film.get('poster_path', None)

    # image manquante
    if poster_path:
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
    else:
        print(f"Image manquante pour le film : {film_title}")
        poster_url = "https://via.placeholder.com/150x225.png?text=Image+Manquante&bg=transparent"

    # Clé unique sur 'id'
    unique_key = f"film_{categorie}_{film_id}_{idx}"

    content = f"""
    <div style="
        text-align: center; 
        cursor: pointer; 
        display: inline-flex;  
        flex-direction: column;  
        justify-content: flex-start;  
        align-items: center;  
        margin: 8px 0 0 0; /* Ajout d'une marge en haut */
        padding: 0; 
        background-color: transparent;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        width: calc(100% - 16px);
        overflow: hidden;
        line-height: 0;
        position: relative;
    ">
        <a id="{unique_key}" href="#" style="
            display: block; 
            background-color: transparent; 
            line-height: 0;
            width: 100%; 
            padding: 0; 
            margin: 0;
            overflow: hidden;
        ">
            <img src="{poster_url}" 
                 style="
                    width: 100%; 
                    height: 100%; /* Permet à l'image de remplir tout le conteneur */
                    border-radius: 15px;
                    object-fit: cover; /* Cela permet de rogner l'image pour qu'elle remplisse le conteneur sans déformer */
                    vertical-align: bottom;
                    margin: 0;
                    padding: 0;
                    display: block;
                    overflow: hidden; /* Assure qu'il n'y a pas de débordement */
                 "/>
        </a>
        <div style="
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: rgba(0, 0, 0, 0.5);
            padding: 5px;
        ">
            <p style="
                color: white; 
                font-size: 14px; 
                font-weight: bold; 
                margin: 0; 
                padding: 0;
                line-height: 1.2;  
                text-align: center;
            ">        
        </div>
    </div>
    """

    return idx, click_detector(content, key=unique_key)


# Fonction de recherche des films par réalisateur
def films_director(name: str, data: pd.DataFrame = data) -> list:
    data = data[data['director'].str.contains(name, case=False)]
    results = data.sort_values(by=['averageRating', 'numVotes'], ascending=False).head(5)
    return results['tconst'].to_list()

# Fonction de recherche des films par acteur
def films_actor(name: str, data: pd.DataFrame = data) -> list:
    films_acteur1 = data[data['actor_1'].str.contains(name, case=False)]
    films_acteur2 = data[data['actor_2'].str.contains(name, case=False)]
    films_acteur3 = data[data['actor_3'].str.contains(name, case=False)]
    
    combined_data = pd.concat([films_acteur1, films_acteur2, films_acteur3]).drop_duplicates(subset=['title'])
    results = combined_data.sort_values(by=['averageRating', 'numVotes'], ascending=False).head(5)
    return results['tconst'].to_list()

def creer_pipeline(data):
    if 'genre_list' not in data.columns or 'list_actor' not in data.columns:
        raise ValueError("Les colonnes 'genre_list' et 'list_actor' sont nécessaires pour entraîner le pipeline.")

    df_to_fit = data[['director', 'genre_list', 'list_actor', 'numVotes', 'averageRating']]
    
    # Création du pipeline avec le préprocesseur et le KNN
    mlb_transformer = FunctionTransformer(lambda x: MultiLabelBinarizer().fit_transform(x), validate=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ('réalisateur', OneHotEncoder(), ['director']),
            ('genres', mlb_transformer, 'genre_list'),
            ('acteurs', mlb_transformer, 'list_actor'),
            ('numvotes', StandardScaler(), ['numVotes']),
            ('note', 'passthrough', ['averageRating'])
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('knn', NearestNeighbors(n_neighbors=5))
    ])
    
    # Créer et entraîner le pipeline
    pipeline.fit(df_to_fit)
    return pipeline


# Fonction pour rechercher les voisins d'un film
def chercher_voisins_id(id: int, pipeline: Pipeline, data: pd.DataFrame) -> list:
    data_scale = pipeline.named_steps["preprocessor"].transform(data[data['tconst'] == id].drop(columns=['tconst']))
    distances, indices = pipeline.named_steps['knn'].kneighbors(data_scale)
    voisins = data.iloc[indices[0][1:]].copy()  # Exclure le film lui-même
    voisins['Distance'] = distances[0][1:]
    return voisins['tconst'].to_list()

def display_banner():
    if st.session_state.page != "personnage":
        st.markdown(
            """
            <div style="background-color: #000; color: #fff; padding: 10px; text-align: center; font-size: 20px;">
                À la recherche du film parfait ? Laissez-nous vous guider ! 🍿
            </div>
            """,
            unsafe_allow_html=True
        )

