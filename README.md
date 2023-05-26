# BDA_Project

## Sujet: Recommandation de musique à partir du dataset Audioscrobbler

### Contexte
Les moteurs de recommandation, dès bien présents sur internet, tels que ceux d’Amazon ou de Spotify, sont reconnus pour leur performance mais surtout leur accessibilité et leur intuitivité d’utilisation. Le projet s’inscrit dans cette tendance et conduira au développement d’un moteur de recommandation de musiques basé sur des méthodes statistiques, Scala et MLlibs. 

### Dataset
Pour commencer, nous sommes partis d'un dataset publié par Audioscrobbler, le moteur de recommandation de last.fm, l’un des principaux sites de radio. D’après leur README, le dataset serait en continuel augmentation. Cependant, nous n’avons accès qu’à une version antérieure, datant du 6 mai 2005, et très limitée en termes de catégories de données. Ainsi, nous avons décidé de partir sur un autre dataset opensource, dénommé "the Million Song Dataset". Le dataset est constitué d'un million d'échantillons d'analyses de chansons, ce qui représente une taille totale de 280 Go. Le jeu de données contient les catégories suivantes:

field name	Type	Description	Link
artist 7digitalid	int	ID from 7digital.com or -1	url
artist hotttnesss	float	algorithmic estimation	url
artist id	string	Echo Nest ID	url
artist latitude	float	latitude	
artist location	string	location name	
artist longitude	float	longitude	
artist mbid	string	ID from musicbrainz.org	url
artist name	string	artist name	url
beats confidence	array float	confidence measure	url
beats start	array float	result of beat tracking	url
duration	float	in seconds
energy	float	energy from listener point of view	
key	int	key the song is in	url
key confidence	float	confidence measure	url
loudness	float	overall loudness in dB	url
mode	int	major or minor	url
mode confidence	float	confidence measure	url
release	string	album name	
similar artists	array string	Echo Nest artist IDs (sim. algo. unpublished)	url
song hotttnesss	float	algorithmic estimation	
tempo	float	estimated tempo in BPM	url
time signature	int	estimate of number of beats per bar, e.g. 4	url
time signature confidence	float	confidence measure	url
title	string	song title	
year	int	song release year from MusicBrainz or 0	url

En phase de développement, nous commencerons par utiliser un sous-ensemble de 10 000 chansons (1%, 1,8 Go compressé). Puis, nous testerons nos modèles sur le dataset complet sur gpu. 

### Questions
Dans le cadre du projet, nous chercherons à répondre à 4 questions: 
- quel(s) sont les genres les plus populaires par année (ranking top n) et les plus populaires au total ?
- quelle est l'année qui comptabilise le plus de chansons produites ?
- quel est le classement des pays selon leur nombre d'artistes (ranking)
- quel est le niveau sonore et le BPM moyen par genre musical ?
