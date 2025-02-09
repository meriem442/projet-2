import streamlit as st
import pandas as pd
import json
import requests
from tools_app import get_clicked
from tools_app import trouver_id, films_actor, films_director
from datetime import datetime
import base64
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from io import BytesIO

st.set_page_config(page_title="Ciné-Explorer", layout="wide")

st.markdown("""
<script>
    // Fonction qui fait défiler la page vers le haut
    function scrollToTop() {
        window.scrollTo(0, 0);  // Force le défilement vers le haut
    }

    // Exécution de la fonction dès que la page est chargée ou mise à jour
    window.onload = scrollToTop;
    window.onbeforeunload = scrollToTop;
</script>
""", unsafe_allow_html=True)

# Image fond écran :
@st.cache_data
def load_image_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode()
    else:
        raise Exception(f"Error downloading image: {response.status_code}")

# GitHub URL for the background image
github_url = "https://github.com/Nathlake/Cine-explorer/raw/main/image_path.jpg"

try:
    image_base64 = load_image_from_github(github_url)
except Exception as e:
    st.error(f"Error loading background image: {e}")

# Personnalisation CSS :
st.markdown(
    f"""
    <style>
    body {{
        background-image: url('data:image/jpeg;base64,{image_base64}');
        background-size: cover;
        background-position: center;
        color: transparent !important; /* Couleur du texte */
    }}
    .stApp {{
        background-color: rgba(0, 0, 0, 0.5) !important; /* Fond semi-transparent sur les composants */
    }}
    /* Personnalisation des titres Streamlit */
    .stHeader, .stSubheader, .stTitle {{
        color: white !important;
        background-color: transparent !important;
    }}
    /* Couleurs spécifiques pour chaque genre */
    .top-10-drama {{ color: #1E90FF !important; }}  /* Bleu pour Drama */
    .top-10-comedy {{ color: #F9E400 !important; }}
    .top-10-animation {{ color: #FFAF00 !important; }}
    .top-10-action {{ color: #71E254 !important; }}
    .top-10-romance {{ color: #F372D0 !important; }}
    .top-10-crime {{ color: #F5004F !important; }}
    
    /* Personnalisation des labels pour les listes déroulantes et sliders */
    .stMultiSelect label, .stSlider label {{
        color: white !important;
        font-size: 20px !important;
        font-weight: bold !important;
        background-color: transparent !important;
    }}
    
    /* Personnalisation des composants de Slider */
    .stSlider > div > div > input {{
        background-color: transparent !important;
        color: white !important;
        border-color: transparent !important;
    }}
    
    .stSlider > div > div > span {{
        color: white !important;
    }}

    /* Personnalisation des boutons Streamlit */
    .stButton > button {{
        background-color: #444444 !important;
        color: white !important;
    }}

    /* Personnalisation des autres composants */
    .stSlider > div > div > input,
    .stTextInput > div > input,
    .stTextArea > div > textarea,
    .stSelectbox div > div > div {{
        background-color: #444444 !important;
        color: white !important;
        border-color: #555555 !important;
    }}

    .stSelectbox div[role="listbox"] {{
        background-color: #444444 !important;
        color: white !important;
    }}

    .stSelectbox div[role="option"] {{
        background-color: #555555 !important;
        color: white !important;
    }}

    .stSelectbox div[role="option"]:hover {{
        background-color: #666666 !important;
    }}

    /* Personnalisation des titres dans la page de détails */
    .stTitle {{
        color: white !important;
        background-color: transparent !important;
        font-size: 30px !important;
    }}

    .stSubheader, .stHeader {{
        color: white !important;
        background-color: transparent !important;
        font-size: 24px !important;
    }}

    /* Personnalisation des sous-titres personnalisés */
    .custom-subtitle {{
        font-size: 24px !important;
        font-weight: bold !important;
        color: white !important; /* S'assurer que les sous-titres sont en blanc */
        background-color: transparent !important;
        padding-top: 10px;
        padding-bottom: 10px;
    }}

    .custom-title {{
        font-size: 35px !important;
        font-weight: bold !important;
        color: white !important; /* S'assurer que les titres sont en blanc */
        background-color: transparent !important;
    }}

    .custom-info {{
        font-size: 18px !important;
        color: white !important; /* S'assurer que l'information est en blanc */
        background-color: transparent !important;
        padding-top: 5px;
        padding-bottom: 5px;
    }}

    /* Personnalisation du texte général */
    .stText {{
        color: white !important; 
    }}

   </style>
   """,
   unsafe_allow_html=True
)

# Chargement data :
@st.cache_data
def load_parquet_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_parquet(BytesIO(response.content))
    else:
        raise Exception(f"Erreur lors du téléchargement du fichier Parquet: {response.status_code}")

@st.cache_data
def load_json_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.content)
    else:
        raise Exception(f"Erreur lors du téléchargement du fichier JSON: {response.status_code}")

# URLs GitHub pour les fichiers
parquet_url = "C:\Users\nemri\OneDrive\Bureau\projet 2\projet 2\df_ready.parquet"
json_url = "C:\Users\nemri\OneDrive\Bureau\projet 2\projet 2\movie_data_with_videos.json"

try:
    data = load_parquet_from_github(parquet_url)
    print(data.head())  # Affiche les premières lignes du dataframe
except Exception as e:
    print(f"Erreur lors du chargement du fichier Parquet : {e}")

try:
    movie_data = load_json_from_github(json_url)
except Exception as e:
    print(f"Erreur lors du chargement du fichier JSON : {e}")

with open('dict_voisins.json', 'r') as f:
    dict_voisins = json.load(f)
dict_voisins = {int(key): value for key, value in dict_voisins.items()}

films_list = data['title'].tolist()

DEFAULT_IMAGE_URL = "https://via.placeholder.com/150x225.png?text=Image+non+disponible"

# Filtrage Top 10 films genres par année :
@st.cache_data
def filtrer_par_annee(data, annee):
    if annee:
        annee_min, annee_max = annee
        return data[(data['startYear'] >= annee_min) & (data['startYear'] <= annee_max)]
    return data

# Récup images API TMDB :
@st.cache_data
def get_poster_url(film_id):
    for movie in movie_data:
        if movie['id'] == film_id:
            poster_path = movie.get('poster_path', '')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return DEFAULT_IMAGE_URL

# Page Accueil :
def afficher_films(categorie, annee=None):
    genre_color_class = {
        "Drama": "top-10-drama",
        "Comedy": "top-10-comedy",
        "Animation": "top-10-animation",
        "Action": "top-10-action",
        "Romance": "top-10-romance",
        "Crime": "top-10-crime"
    }

    if annee is None and "annee" in st.session_state:
        annee = st.session_state.annee

    genre_class = genre_color_class.get(categorie, "")
    
    if st.session_state.page == "accueil":
        st.markdown(
            f'<div class="{genre_class}" style="font-size: 24px; font-weight: bold; margin-bottom: 15px;">Top 10 {categorie}</div>',  # Réduit la marge à 15px
            unsafe_allow_html=True
        )

        cols = st.columns(10)
        
        st.markdown(
            f"""
            <style>
                /* Retire toute marge/padding globale pour les colonnes */
                .css-1v3fvcr, .stColumn, .css-16hu7z2 {{
                    padding: 0 !important;
                    margin: 0 !important;
                    background: transparent !important;
                    border: none !important;
                }}

                /* Retirer tout fond blanc ou toute bordure */
                .stMarkdown {{
                    margin: 0 !important;
                    padding: 0 !important;
                    background: transparent !important;
                }}
                
                /* Définir les colonnes à un fond transparent */
                .css-1x8thse {{
                    background-color: transparent !important;
                    padding: 0 !important;
                }}
                
                /* Retirer toute bordure autour des images */
                img {{
                    border: none !important;
                    box-shadow: none !important;
                    margin: 0 !important;
                    padding: 0 !important;
                }}
            </style>
            """, unsafe_allow_html=True
        )
        
        # Filtre des films en fonction du genre et de l'année
        genre_films = data[data['genres'].str.contains(categorie, case=False)]
        
        if annee:
            genre_films = filtrer_par_annee(genre_films, annee)
        
        genre_films = genre_films[genre_films['numVotes'] > 100000]
        genre_films_sorted = genre_films.sort_values(by=['averageRating', 'numVotes'], ascending=False).head(10)

        for idx in range(len(genre_films_sorted)):
            col = cols[idx]
            film = genre_films_sorted.iloc[idx]
            film_title = film['title']
            film_id = film["id"]

            film_data = trouver_id(film_id)

            if film_data is not None:
                film_id = film_data.get("id", None)
                film_title = film_data.get("title", "Titre inconnu")
                poster_path = film_data.get('poster_path', None)
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            
            unique_key = f"film_{categorie}_{film_id}_{idx}"

            with col:
                clicked = get_clicked(
                    movie_data,
                    [film_title],
                    film_id,
                    categorie,
                    key_=True
                )[1]
                
                if clicked:
                    st.session_state.selected_film = film_title
                    st.session_state.page = "details"
                    st.rerun()

            # Ajouter titre du film en dessous de l'image
                st.markdown(f"""
        <div style='display: flex; justify-content: center; width: 100%;'>
            <p style='
                text-align: center; 
                font-size: 16px; 
                margin-top: 5px; 
                font-weight: bold; 
                max-width: calc(100% - 16px);
                word-wrap: break-word;
                line-height: 1.2;
            '>
                {film_title}
            </p>
        </div>
    """, unsafe_allow_html=True)

                for _ in range(3):  # Ajoute trois sauts de ligne pour plus d'espacement entre les films
                    st.markdown("<br>", unsafe_allow_html=True)


                    
def page_accueil():
    st.markdown("""
    <style>
    /* Pour les images */
    .stImage, .stImage > div, .stImage img {
        background-color: transparent !important; /* Supprime le fond de l'image */
        margin: 0 !important;  /* Supprime toute marge autour de l'image */
        padding: 0 !important;  /* Supprime tout padding autour de l'image */
        border: none !important; /* Supprime les bordures autour de l'image */
    }
    
    /* Applique une transparence totale sur les titres et autres éléments textuels */
    .custom-title, .custom-subtitle, .custom-info, p, h1, h2, h3, h4, h5, h6, i {
        color: white !important;  /* Force le texte en blanc */
        background: none !important;  /* Supprime tout fond derrière les textes */
        margin: 0 !important;  /* Supprime toutes les marges */
        padding: 0 !important;  /* Supprime tous les paddings */
    }

    /* Forcer les colonnes à être sans bordure et sans fond */
    .stColumn {
        background-color: transparent !important;  /* Retirer le fond des colonnes */
        padding: 0 !important;  /* Supprimer toute marge/padding dans les colonnes */
        margin: 0 !important;  /* Supprimer toute marge dans les colonnes */
    }

    /* Forcer l'alignement et éviter les espaces indésirables autour des titres */
    .stMarkdown {
        background: transparent !important;  /* Assurer que le fond soit transparent */
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Empêcher les éléments de Streamlit d'afficher un fond derrière les titres ou images */
    .css-1v3fvcr {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

    selected_film = st.multiselect("Choisissez un film et obtenez des recommandations personnalisées 🎦", films_list, placeholder="Entrez ou sélectionnez le nom d'un film...")

    annee_min = int(data['startYear'].min())
    annee_max = int(data['startYear'].max())
    
    # Si une année a été sélectionnée précédemment => valeur par défaut
    annee_par_defaut = st.session_state.get('annee', (annee_min, annee_max))  # Utilise la sélection précédente ou les valeurs minimales et maximales par défaut
    
    annee = st.slider(
        "Filtrer les Top 10 par année",
        min_value=annee_min,
        max_value=annee_max,
        value=annee_par_defaut,  # Valeur par défaut restaurée
        step=1
    )
    
    # Sauvegarder la sélection de l'année dans session_state
    st.session_state.annee = annee

    # Afficher les films en fonction de la sélection de l'année
    afficher_films("Drama", annee=annee)
    afficher_films("Comedy", annee=annee)
    afficher_films("Animation", annee=annee)
    afficher_films("Action", annee=annee)
    afficher_films("Romance", annee=annee)
    afficher_films("Crime", annee=annee)
    
    # vers la page de détails
    if selected_film:
        st.session_state.selected_film = selected_film[0]
        st.session_state.page = "details"
        st.rerun()

# Obtenir les films similaires à un film sélectionné
def get_similar_films(selected_info, num_films=5):
    genres = selected_info.get('genres', [])
    start_year = selected_info.get('startYear', 0)

    similar_films = data[
        (data['startYear'] == start_year) & 
        (data['genres'].apply(lambda x: any(genre in x for genre in genres)))
    ]
    
    # Tri des films similaires par note moyenne, décroissante
    similar_films_sorted = similar_films.sort_values(by='averageRating', ascending=False)
    return similar_films_sorted.head(num_films)


import requests

def is_valid_image_url(url):
    """Vérifie si l'URL de l'image est valide en envoyant une requête HTTP GET."""
    try:
        # Envoyer une requête pour vérifier que l'URL de l'image renvoie un code HTTP 200
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # Si une exception est levée, l'URL est considérée comme invalide
        return False


import requests
@st.cache_data
def get_film_poster_url(film_id):
    api_key = st.secrets["tmdb_api_key"]
    base_url = "https://api.themoviedb.org/3/movie/"
    
    response = requests.get(f"{base_url}{film_id}?api_key={api_key}&language=fr")
    
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return poster_url
    return "https://via.placeholder.com/150x225.png?text=Image+non+disponible"  

def page_details():
    if st.button("Retour", key="back_to_home"):
        st.session_state.page = "accueil"
        st.rerun()

    selected_info = data[data['title'] == st.session_state.selected_film].iloc[0]

    if not selected_info.empty:
        # Données des acteurs
        actors_data = []
        for i in range(1, 4):
            actor_name = selected_info.get(f'actor_{i}', None)
            actor_profile_path = selected_info.get(f'actor_{i}_profile_path', None)
            actor_character = selected_info.get(f'actor_{i}_character', None)
            actors_data.append((actor_name, actor_profile_path, actor_character))

        poster_url_high_res = get_poster_url(selected_info['id']).replace('w185', 'original')  

        # Poster_url comme fond d'écran pour la page de détails
        st.markdown(
            f"""
            <style>
            .background-container {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-image: url("{poster_url_high_res}");
                background-size: cover;
                background-position: center center;
                z-index: -1;  /* Placer l'image en arrière-plan */
                opacity: 0.9;  /* Moins de transparence que 0.7, image plus visible */
            }}
            .stApp {{
                background-color: rgba(0, 0, 0, 0.5) !important; /* Fond semi-transparent pour les composants */
            }}
            .stTitle, .stSubheader, .stHeader {{
                background-color: transparent !important;
                color: white !important;
            }}
            </style>
            <div class="background-container"></div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([3, 5, 3])

        with col1:
            st.markdown(
                f'<img src="{poster_url_high_res}" width="350" style="border-radius: 20px; border: 2px solid #555555; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);" />',
                unsafe_allow_html=True
            )

        with col2:
            # Titre du film
            start_year = selected_info.get('startYear', 'Année non disponible')
            st.markdown(f"<h2 class='custom-title' style='color: white; text-align: left;'>{selected_info['title']} ({start_year})</h2>", unsafe_allow_html=True)

            # Genres du film
            genres = selected_info.get('genres', [])
            if genres:
                genre_list = genres if isinstance(genres, list) else genres.split(',')
                st.markdown(f"<h3 class='custom-subtitle' style='color: white;'>Genre(s) : {', '.join(genre_list)}</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 class='custom-subtitle' style='color: white;'>Genre(s) : Non disponible</h3>", unsafe_allow_html=True)

            # Note et nombre de votes
            average_rating = selected_info.get('averageRating', 'Note non disponible')
            st.markdown(f"<p class='custom-info' style='color: white;'>Note IMDB ⭐ {average_rating}</p>", unsafe_allow_html=True)

            numVotes = selected_info.get('numVotes', 0)
            formatted_votes = f"{numVotes / 1_000_000:.1f}M" if numVotes >= 1_000_000 else f"{numVotes / 1_000:.1f}K" if numVotes >= 1_000 else str(numVotes)
            st.markdown(f"<p class='custom-info' style='color: white;'>Total votes 👍 {formatted_votes} </p>", unsafe_allow_html=True)

            # Réalisateur
            director_name = selected_info.get('director', 'Réalisateur non disponible')
            director_profile_path = selected_info.get('director_profile_path', None)
            
            if director_name != 'Réalisateur non disponible' and director_profile_path:
                director_image_url = f"https://image.tmdb.org/t/p/original{director_profile_path}"
                
                # Vérification de l'URL de l'image avant de l'afficher
                if not is_valid_image_url(director_image_url):
                    director_image_url = "https://via.placeholder.com/150x225.png?text=Photo+non+disponible"
                
                st.markdown(f"<h3 class='custom-subtitle' style='color: white;'>Réalisateur :</h3>", unsafe_allow_html=True)
                st.markdown(f'<img src="{director_image_url}" width="150" style="border-radius: 20px; border: 2px solid #555555; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);" />', unsafe_allow_html=True)
                st.markdown(f"<h3 class='custom-subtitle' style='color: white;'>{director_name}</h3>", unsafe_allow_html=True)

                # Bouton "En savoir plus"
                if st.button(f"En savoir plus", key=f"director_{selected_info['id']}"):
                    st.session_state.selected_person = director_name
                    st.session_state.page = "director_details"
                    st.rerun()
            else:
                # Afficher l'image de remplacement si pas d'image de réalisateur
                st.markdown('<img src="https://via.placeholder.com/150x225.png?text=Photo+non+disponible" width="150" style="border-radius: 20px; border: 2px solid #555555; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);" />', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h3 class='custom-subtitle' style='color: white;'>Acteurs principaux :</h3>", unsafe_allow_html=True)

            # Liste des acteurs filtrée
            actors = []
            for i in range(1, 4):
                actor_name = selected_info.get(f'actor_{i}', None)
                actor_profile_path = selected_info.get(f'actor_{i}_profile_path', None)
                actor_character = selected_info.get(f'actor_{i}_character', None)

                if actor_name and actor_profile_path and actor_name != "Non disponible":
                    actors.append((actor_name, actor_profile_path, actor_character))

            # Si des acteurs existent (Non disponibles exclus), on les affiche
            if actors:
                actor_columns = st.columns(len(actors), gap="small")
                for idx, (actor_name, actor_profile_path, actor_character) in enumerate(actors):
                    with actor_columns[idx]:
                        # Si aucune image de profil n'est fournie, on affiche l'image de remplacement
                        if not actor_profile_path or actor_profile_path == "null":
                            actor_image_url = "https://via.placeholder.com/150x225.png?text=Photo+non+disponible"
                        else:
                            actor_image_url = f"https://image.tmdb.org/t/p/original{actor_profile_path}"

                        # Vérification de l'URL de l'image avant de l'afficher
                        if not is_valid_image_url(actor_image_url):
                            actor_image_url = "https://via.placeholder.com/150x225.png?text=Photo+non+disponible"
                        
                        st.markdown(
                            f'<img src="{actor_image_url}" width="150" style="border-radius: 20px; border: 2px solid #555555; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);" />',
                            unsafe_allow_html=True
                        )

                        # Affichage du nom et du personnage joué
                        if actor_character:
                            actor_character = str(actor_character).replace('"', '').strip("[]").replace("'", "")
                            st.markdown(f"<h4 class='custom-subtitle' style='color: white;'>{actor_name}</h4>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: white;'><i>{actor_character}</i></p>", unsafe_allow_html=True)

                        button_key = f"actor_{selected_info['id']}_{idx}"
                        if st.button(f"En savoir plus", key=button_key):
                            st.session_state.selected_person = actor_name
                            st.session_state.page = "actor_details"
                            st.rerun()

        with col3:
            # Bande annonce
            st.markdown('<div class="custom-subtitle" style="color: white;">Bande annonce :</div>', unsafe_allow_html=True)
            id = selected_info['id']
            video_data = next((movie for movie in movie_data if movie['id'] == id), None)
            if video_data and 'videos' in video_data:
                videos = video_data['videos']
                trailer = next((video for video in videos if video['type'] == 'Trailer'), None)
                if trailer:
                    st.markdown(f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{trailer["key"]}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
                else:
                    st.write("Aucune bande-annonce disponible pour ce film.")
            else:
                st.write("Aucune vidéo trouvée pour ce film.")

            # Synopsis
            st.markdown('<div class="custom-subtitle" style="color: white;">Synopsis :</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="text-align: justify; color: white;">{selected_info.get("overview", "Synopsis non disponible.")}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='custom-subtitle' style='color: white;'>Films similaires :</h3>", unsafe_allow_html=True)
    
        id_film = selected_info['id']
        films_similaires = dict_voisins[id_film] # Fichier dict_voisins (machine learning)
        film_titles_dict = dict(zip(data['id'], data['title']))

        # Vérification si films similaires contient des film_ids valides
        if films_similaires:
            movie_columns = st.columns(min(5, len(films_similaires))) 

        for idx, film_id in enumerate(films_similaires):
            film_title = film_titles_dict.get(film_id, "Titre inconnu")
            film_poster_url = get_film_poster_url(film_id) 

            col_idx = idx % len(movie_columns)

            with movie_columns[col_idx]:
                # Affichage de l'affiche du film
                st.markdown(
                    f'<img src="{film_poster_url}" width="150" height="225" style="border-radius: 10px; border: 2px solid #555555; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);" />',
                    unsafe_allow_html=True
                )

                # Affichage du titre du film
                st.markdown(f"<p style='color: white; text-align: center; font-weight: bold; word-wrap: break-word; max-width: 150px;'>{film_title}</p>", unsafe_allow_html=True)

                # Bouton de détails pour chaque film
                if st.button(f"Détails", key=f"details_{film_id}"):
                    st.session_state.selected_film = film_title  # Enregistrer le film sélectionné
                    st.session_state.page = "details"  # Aller à la page de détails
                    st.rerun()  # Rafraîchir la page pour afficher la page de détails

# Page Réalisateur, acteur (après sélection)
st.markdown(
    """
    <style>
    /* Justifier les paragraphes et les blocs de texte */
    .custom-text, p, div {
        text-align: justify !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def page_personnage():
    selected_person = st.session_state.selected_person  # Acteur ou réalisateur sélectionné

    # Dictionnaire tconst + titres films
    film_titles_dict = dict(zip(data['tconst'], data['title']))

    person_info = None

    # Recherche acteur ou réalisateur
    if selected_person in data['director'].values:
        # Si c'est un réalisateur :
        person_info = data[data['director'] == selected_person].iloc[0]
        name = person_info['director']
        birthday = person_info['director_birthday']
        biography = person_info['director_biography']
        if pd.isna(biography) or biography == '':
            biography = "Biographie non disponible"
        profile_path = person_info['director_profile_path']
        filmography = films_director(name)
        person_type = "director"
            
    else:
        actor_columns = ['actor_1', 'actor_2', 'actor_3']
        for col in actor_columns:
            if selected_person in data[col].values:
                # Si c'est un acteur :
                person_info = data[data[col] == selected_person].iloc[0]
                name = person_info[col]
                birthday = person_info[f'{col}_birthday']
                biography = person_info[f'{col}_biography']
                if pd.isna(biography) or biography == '':
                    biography = "Biographie non disponible"
                profile_path = person_info[f'{col}_profile_path']
                filmography = films_actor(name)
                person_type = f"actor_{actor_columns.index(col) + 1}"
                break
    
    if person_info is None:
        st.write("Informations non disponibles.")
        return

    # Enregistre la personne sélectionnée dans la session pour y revenir plus tard
    st.session_state.selected_person_info = person_info

    col1, col2 = st.columns([1, 9])

    with col1:
        if st.button("🏠", key="home_button_person"):
            st.session_state.page = "accueil"
            st.rerun()

        if st.button("Retour", key="back_button_person"):
            st.session_state.page = "details"
            st.rerun()

    # Contenu de la page personnage
    col1, col2 = st.columns([3, 7])

    with col1:
        # Image de profil :
        if profile_path:
            person_image_url = f"https://image.tmdb.org/t/p/w500{profile_path}"
        else:
            person_image_url = 'https://via.placeholder.com/150x225.png?text=Photo+non+disponible'
        
        st.markdown(
            f'<img src="{person_image_url}" width="300" height="450" style="border-radius: 20px; border: 2px solid #555555; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);" />',
            unsafe_allow_html=True
        )

    with col2:
        # Nom de la personne :
        st.markdown(f"<h2 style='color: white;'>{name}</h2>", unsafe_allow_html=True)

        # Date de naissance :
        if birthday:
            try:
                # Format date de naissance :
                birthday = datetime.strptime(birthday, "%Y-%m-%d").strftime("%d-%m-%Y")
            except ValueError:
                birthday = "non disponible"

        st.markdown(f"<p style='color: white;'>Date de naissance : {birthday}</p>", unsafe_allow_html=True)

        # Biographie :
        biography_paragraphs = biography.split('\n')  
        for paragraph in biography_paragraphs:
            st.markdown(f"<p style='color: white;'>{paragraph}</p></p>", unsafe_allow_html=True)

        # Films joués/réalisés :
        st.markdown("<h3 style='color: white;'>Filmographie :</h3>", unsafe_allow_html=True)

        # Vérification si filmography contient des film_ids valides
        if filmography:
            # Créer des colonnes en fonction du nombre de films
            movie_columns = st.columns(min(5, len(filmography)))  
            for idx, film_id in enumerate(filmography):
                film_title = film_titles_dict.get(film_id, "Titre inconnu")  
                film_poster_url = get_film_poster_url(film_id)  

                col_idx = idx % len(movie_columns)  

                with movie_columns[col_idx]:
                    # Affichage de l'affiche du film
                    st.markdown(
                        f'<img src="{film_poster_url}" width="150" height="225" style="border-radius: 10px; border: 2px solid #555555; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);" />',
                        unsafe_allow_html=True
                    )

                    # Affichage du titre du film
                    st.markdown(f"<p style='color: white; text-align: center; font-weight: bold; word-wrap: break-word; max-width: 150px;'>{film_title}</p>", unsafe_allow_html=True)

                    # Bouton de détails pour chaque film
                    if st.button(f"Détails", key=f"details_{film_id}"):
                        st.session_state.selected_film = film_title  # Enregistrer le film sélectionné
                        st.session_state.page = "details"  # Aller à la page de détails
                        st.rerun()  # Rafraîchir la page pour afficher la page de détails
        else:
            st.write("Aucun film trouvé dans la filmographie.")


# Titres, boutons retour/accueil, note bas de page

if "page" not in st.session_state:
    st.session_state.page = "accueil"

def header_with_back_button():
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🏠", key="back_button"):
            st.session_state.page = "accueil"
            st.rerun()
    with col2:
        if st.session_state.page == "accueil":
            st.markdown("<h1 class='custom-header' style='color:transparent;'>À la recherche du film parfait ? Laissez-nous vous guider ! 🍿</h1>", unsafe_allow_html=True)

if st.session_state.page == "accueil":
    header_with_back_button()
    page_accueil()
    st.markdown(
    """
    <div style="
        display: flex;
        justify-content: flex-end;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
        color: white;
    ">
        🎥 Les réalisateurs pour ce projet : Benjamin, Benoit, Meriem & Nathalie
    </div>
    """,
    unsafe_allow_html=True
)
    url1 = "https://www.imdb.com/fr/?ref_=nv_home"
    url2 = "https://www.themoviedb.org/"
    st.markdown("Sources : [imdb](%s) | [tmdb](%s)" % (url1, url2))

elif st.session_state.page == "details":
    header_with_back_button()  
    page_details()

if 'page' not in st.session_state:
    st.session_state.page = 'film_details'  

if st.session_state.page == 'film_details':
    page_details()
elif st.session_state.page == 'actor_details' or st.session_state.page == 'director_details':
    page_personnage()
