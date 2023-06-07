# BDA_Project

## Sujet: Recommandation de musique à partir du dataset Audioscrobbler

### Contexte
Les moteurs de recommandation, dès bien présents sur internet, tels que ceux d’Amazon ou de Spotify, sont reconnus pour leur performance mais surtout leur accessibilité et leur intuitivité d’utilisation. Le projet s’inscrit dans cette tendance et conduira au développement d’un moteur de recommandation de musiques basé sur des méthodes statistiques, Scala et MLlibs. 

### Dataset
Pour commencer, nous sommes partis d'un dataset publié par Audioscrobbler, le moteur de recommandation de last.fm, l’un des principaux sites de radio. D’après leur README, le dataset serait en continuel augmentation. Cependant, nous n’avons accès qu’à une version antérieure, datant du 6 mai 2005, et très limitée en termes de catégories de données. Ainsi, nous avons décidé de partir sur un autre dataset opensource, dénommé "the Million Song Dataset". Le dataset est constitué d'un million d'échantillons d'analyses de chansons, ce qui représente une taille totale de 280 Go. Le jeu de données contient les catégories suivantes:

```
| Field Name               | Type       | Description                              |
|--------------------------|------------|------------------------------------------|
| artist hotttnesss        | float      | algorithmic estimation                   |
| artist id                | string     | Echo Nest ID                             |
| artist latitude          | float      | latitude                                 |
| artist location          | string     | location name                            |
| artist longitude         | float      | longitude                                |
| artist name              | string     | artist name                              |
| beats confidence         | array float| confidence measure                       |
| beats start              | array float| result of beat tracking                  |
| duration                 | float      | in seconds                               |
| energy                   | float      | energy from listener point of view       |
| key                      | int        | key the song is in                       |
| key confidence           | float      | confidence measure                       |
| loudness                 | float      | overall loudness in dB                   |
| mode                     | int        | major or minor                           |
| mode confidence          | float      | confidence measure                       |
| release                  | string     | album name                               |
| similar artists          | array str  | Echo Nest artist IDs (sim. unpublished)  |
| song hotttnesss          | float      | algorithmic estimation                   |
| tempo                    | float      | estimated tempo in BPM                   |
| time signature           | int        | estimate of number of beats per bar      |
| time signature confidence| float      | confidence measure                       |
| title                    | string     | song title                               |
| year                     | int        | song release year from MusicBrainz or 0  |

```

En phase de développement, nous commencerons par utiliser un sous-ensemble de 10 000 chansons (1%, 1,8 Go compressé). Puis, nous testerons nos modèles sur le dataset complet sur gpu. 

### Questions
Dans le cadre du projet, nous chercherons à répondre à 4 questions: 
- quels sont les genres les plus populaires? quelle est l'année qui comptabilise le plus de chansons produites? quel pays détient le plus grand nombre d'artiste?
- quel est le niveau sonore moyen et le BPM moyen (battement par minute) par genre musical ?
- Comment prédire le genre musical d'une musique à partir des caractéristiques d'autres musiques (niveau sonore, tempo, gamme..) -> Machine learning
- Dans une optique de recommandation d'un artiste à un utilisateur, comment pourrait-on mesurer la similarité entre artistes ? -> Machne learning

### Done
- récupérer toutes les données de l'échantillon de 1GB comportant 10000 musiques, en format .h5. le transformer en .csv et enfin en .parquet pour accélérer le flux de données dans spark
- faire de la data augmentation en récupérant le pays d'un artiste selon ses coordonnées lat/long, car la feature "location" est très incohérante; utilisé geopy dans python mais idéalement il faudrait le faire dans spark avec Geospark sauf que je n'arrive pas à uploader la librairie et ses dépendances dans le fichier sbt de notre projet...
- faire de la data augmentation en récupérant le genre de chaque artiste, par des simples requêtes dans l'api lastfm
- créer le projet dans IntelliJ; Dans src/main/scala/files, se trouvera le fichier objet avec toutes nos features et leur type et le fichier pour les query et traitement de données ; dans src/main/data, se trouvera le fichier de données .parquet. 
- fonction pour récupérer les données dans le fichier .parquet
- 3 fonctions pour répondre à la question 1
- 2 fonctions pour répondres à la question 2

### A faire
- multiprocessing, préférablement avec spark, de la data augmentation pour les features "artist_location" et "artist_genre", sinon pour 1M de musiques, nous allons périr
- question 3 et 4 (machine learning)
- illustrer les Q1 Q2 avec des beaux graphiques !
- des libraires scala permettent de faire de la visualisation de données, nous pourrions par example créer une map du monde et appliquer un code couleur représentant la densité d'artistes par pays

### Remarques
- le projet utilise SDK 11
- le code le plus récent se trouve sur la branch main
- toutes les fonctions sont inscrites dans le main, donc juste à runner et elles se lancent toutes. Suffira simplement de faire un .show()
- les données (csv et parquet) sont à ranger localement dans le fichier data prévu à cet effet 
- pour la dataset complet, prévoir une implémentation d'un processus Distributed Computing, sinon ça va prendre 16h.... Apach Spark s'y prête bien mais pas sûr que cela fonctionne sur des tâches de type requête
- le filtrage est bien plus restrictif sur la colonne artiste_terms, que la colonne "artiste_genre".. 

### Modèles
#### Apache Spark MLlib
#### Question 3: RNN on genre prediciton (pour pallier au besoin de passer par l'API)
- Preprocessing:
``` 
             tempo      loudness  time_signature      duration
count  10000.000000  10000.000000    10000.000000  10000.000000
mean     122.915449    -10.485668        3.564800    238.507518
std       35.184412      5.399788        1.266239    114.137514
min        0.000000    -51.643000        0.000000      1.044440
25%       96.965750    -13.163250        3.000000    176.032200
50%      120.161000     -9.380000        4.000000    223.059140
75%      144.013250     -6.532500        4.000000    276.375060
max      262.828000      0.566000        7.000000   1819.767710
```
```
tempo               0
loudness            0
beats_start         0
time_signature      0
duration            0
artist_genre      155
```

- remove music with too high or too low duration
- remove music with time signature = 0
- remove music with tempo = 0
- check ranges of values for each feature
- normaliser avec min/max scaling les features ou autre méthode
- joindre graphe av/ap data transformation

#### Approche:
- Step 1: Load and prepare the data
- Step 2: Split the data into training and testing sets
- Step 3: Define the feature transformation and explain why use it on which feature (outliers...)
    - tempo: normal shaped distrbiution, no outliers => min/max scaling (sensitive to outliers)
    - loudness : negatively skewed, add constant = 100 to have normal distribution, no outliers, min/max scaling
    - time_signature : nothing, no outliers,, form of one-hot encoding
    - duration : positively skewed, no outliers, min_max rescale
- Step 4: Select a supervised learning algorithm (RNN)
- Step 5: Train the model
- Step 6: Make predictions
- Step 7: Evaluate the model
- Step 8: Fine-tune and iterate as needed

