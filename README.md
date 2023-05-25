# BDA_Project

## Sujet: Recommandation de musique à partir du dataset Audioscrobbler

### Contexte
Les moteurs de recommandation, dès bien présents sur internet, tels que ceux d’Amazon ou de Spotify, sont reconnus pour leur performance mais surtout leur accessibilité et leur intuitivité d’utilisation. Le projet s’inscrit dans cette tendance et conduira au développement d’un moteur de recommandation de musiques basé sur des méthodes statistiques, Scala et MLlibs.

### Dataset
Le dataset fût publié par Audioscrobbler, le moteur de recommandation de last.fm, l’un des principaux sites de radio. D’après leur README, le dataset serait en continuel augmentation. Cependant, nous n’avons accès qu’à une version antérieure, datant du 6 mai 2005. Le dataset est constitué de 3 fichiers txt:
- user_artist_data.txt : 3 colonnes : userid artistid playcount
taille: 141,000 users uniques, 1.6 millions d’artistes uniques, 24.2 millions de titres écoutés
- artist_data.txt : 2 colonnes : artist_id, artist_name (id et nom des artistes)
- artist_alias.txt : 2 colonnes : bad_id, good_id (artistes dont l'orthographe est incorrecte et leur identifiant d'artiste correct)

### Questions
Dans le cadre du projet, nous chercherons à répondre à 4 questions: 
Comment recommander des artistes/musiques “susceptibles” d’intéresser un utilisateur à partir des goûts de milliers d’autres utilisateurs et de sa liste de lecture ?
Comment recommander des artistes aux utilisateurs à partir des tendances d’écoutes (artistes/albums les plus écoutés susceptibles d'intéresser un utilisateur)
Quels sont les artistes/albums les plus écoutés/populaires par décennie ?
Comment réaliser un ranking des années selon leur nombre total d’écoutes cumulées ? en déduire la ou les décennies les plus actives

### Périmètre
Notre dataset étant relativement limité en termes de catégorie de données, nous ferons très certainement usage de Data augmentation, pour récupérer par exemple le genre musical propre à chaque artiste.
