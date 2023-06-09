# BDA_Project

## Sujet: Recommandation de musique à partir du dataset the ≈

### Contexte
Les moteurs de recommandation, dès bien présents sur internet, tels que ceux d’Amazon ou de Spotify, sont reconnus pour leur performance mais surtout leur accessibilité et leur intuitivité d’utilisation. Le projet s’inscrit dans cette tendance et conduira au développement d’un moteur de recommandation de musiques basé sur des méthodes statistiques, Scala et MLlibs. 

### Description du dataset
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

Nos modèles seront entrainés sur un échantillon de 10 000 chansons. Les données seront récupérées depuis un échantillon de 1GB comportant 10000 musiques et présentant une grande diversité, en format .h5. Elles seront ensuite transformées en .csv et enfin en .parquet.

### Description des features utilisées et data augmentation 

Pour le clustering d'artiste, nous avons utilisé les features suivantes:   
```
|artist_id: String,
|artist_name: String,
|artist_location: String,
|artist_latitude: Float,
|artist_longitude: Float,
|nbSong : Int,
|avgSongDuration: Float, 
|avgSongLoudness: Float,
|avgTempo: Float,
|yearFirstSong: Int,
|yearLastSong: Int,
|avgLoudness: Float,
|avgEnergy: Float
```

Pour le clustering sur genre musical, nous avons utilisé les features suivantes:
```
|time_signature = Int
|time_signature_confidence = Float
|key = Int
|key_confidence = Float
|loudness = Float
|mode = Int
|mode_confidence = Float
|tempo = Float
|year = Int
```

Pour la classification supervisée sur le genre muscial (multiple layer perceptron, decision tree, random forest), nous avons utilisé les features suivantes:
```
|loudness = Float
|tempo = Float
|duration = Float
|time_signature = Int
```

Afin de répondre aux deux premières questions, nous avons fait usage de data augmentation pour générer la localité des artistes à partir de leurs coordonnées géographique en utilisant la librairie Geopy. 

### Questions

Dans le cadre du projet, nous chercherons à répondre à 4 questions: 
- Question 1: Quels sont les genres les plus populaires? Quelle est l'année qui comptabilise le plus de chansons produites? Quel pays détient le plus grand nombre d'artiste?
- Question 2: Quel est le niveau sonore moyen et le BPM moyen (battement par minute) par genre musical ?
- Question 3: Comment prédire le genre musical d'une musique à partir des caractéristiques d'autres musiques (niveau sonore, tempo, gamme, durée) -> Machine learning
- Question 4: Dans une optique de recommandation d'un artiste à un utilisateur, comment pourrait-on mesurer la similarité entre artistes ? -> Machine learning

### Data preprocessing

Pour la classification de la question 3, plusieurs méthodes de preprocessing ont été appliquées sur les données. Voici l'état initial des features avant le preprocessing:
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

Les méthodes qui ont été appliquées sont:
- retirer les musiques avec une durée trop élevée ou trop faible
- retirer les musiques avec une signature temporelle (time signature) égale à zéro 
- retirer les musiques avec un tempo égal à zéro
- normaliser les caractéristiques avec une méthode telle que la mise à l'échelle min-max 

Anto:  vos méthodes de preprocessing

Dalia/AS:  vos méthodes de preprocessing

### Algorithmes

La question 3 visera à prédire le genre musical d'une musique à partir de ses caractéristiques (niveau sonore, tempo, gamme, durée), de manière supervisée, en utiliseant des algorithmes de machine learning tels que le Decision Tree (arbre de décision), le Random Forest et le Multi-Layer Perceptron (MLP). Le Random Forest est robuste face aux données bruyantes et évite le surajustement en sélectionnant des sous-ensembles aléatoires de caractéristiques et de données pour chaque arbre de décision. Les résultats le confirment, avec un score d'accuracy supérieur aux deux autres.

L'approche suivie se compose de 8 étapes:

- Étape 1: Charger et préparer les données
Utiliser les genres musicaux avec la plus grande densité (rock, pop, etc.) comme réponse Y. Cela permettra au modèle de mieux généraliser en réduisant la dimensionnalité.
- Étape 2: Diviser les données en ensembles d'entraînement et de test.
- Étape 3: Définir la transformation des caractéristiques et expliquer pourquoi ?
    - Tempo : Distribution de forme normale, pas de valeurs aberrantes => mise à l'échelle min/max (sensible aux valeurs aberrantes).
    - Loudness : Distribution à asymétrie négative, ajouter une constante = 100 pour obtenir une distribution normale, pas de valeurs aberrantes, mise à l'échelle min/max.
    - Time_signature : Pas de traitement nécessaire, pas de valeurs aberrantes, forme de codage one-hot.
    - Durée : Distribution à asymétrie positive, pas de valeurs aberrantes, mise à l'échelle min/max.
- Étape 4: Sélectionner un algorithme d'apprentissage supervisé: decision tree, random forest, MLP
- Étape 5: Entraîner le modèle.
- Étape 6: Faire des prédictions.
- Étape 7: Évaluer le modèle.
- Étape 8: Fine tuning sur les hyperparamètres: GridSEarch + cross validation 5 fold
````
val paramGrid = new ParamGridBuilder()
    .addGrid(dt.maxDepth, Array(5, 10, 15))
    .addGrid(dt.maxBins, Array(16, 32, 48))
    .addGrid(dt.numTrees, Array(50, 100, 200))
    .build()
val crossValidator = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(5)
`````

- Etape 9: Comparer les résultats aux autres modèles


Anto: La question 3 cherchera également à prédire des musiques par une technique de clustering.....

Dalia / AS: La question 4 visera à trouver, quant à elle, une relation de similarité entre artistes, également par une technique de clustering.....

### Optimisation


### Results

**Question 1**

- Quels sont les genres les plus populaires ?

![test](./img/results_q1_part3.png)

- Quel pays détient le plus grand nombre d'artiste ?

![test](./img/results_q1_part1.png)

- Quelle est l'année qui comptabilise le plus de chansons produites ?

![test](./img/results_q1_part2.png)


**Question 2**

// TODO

**Question 3**

-  Classification on artiste_genre: Model fine tuning results
![test](./img/results_q3_part1.jpeg)

Random Forest: The accuracy ranges from 43.00% to 44.00%, with the highest accuracy achieved at 200 trees. The accuracy is relatively consistent across different max depths.
Decision Tree: The accuracy ranges from 44.70% to 38.6%, with the highest accuracy achieved at a max depth of 5. Increasing the max depth to 20 leads to a decrease in accuracy.
MLP: The accuracy ranges from 42.40% to 41.1%. It seems that increasing the number of layers or iterations does not necessarily lead to higher accuracy. The choice of solver and step size can also affect accuracy, but not in our case.
Overall, the Random Forest model performs slightly better than the Decision Tree and MLP models based on the provided accuracy results. 

We also added a gridSearch to find the optimal pair of hyperparameter that maximizes accuracy. Here are the results:
```
Best Model Parameters:
maxBins: 50
impurity: gini
maxDepth: 5
subsamplingRate: 0.8
numTrees: 200
```
And finally performed a cross validation from the optimal set of parameters found previously to also maximize the accuracy of our model shuffling 5 fold our dataset. Here's the final result: 4280228556034745.

Overall, the gridSearch et the cross validation didn't improve much the accuracy of our model. As a matter of fact, all models seem to output simiar accuracies so the problem may lie inside the data lebeling itself, and since we do not know how the dataset was labeled, it will be difficult to find any correlation.

**Question 4**

// TODO

### Possible future enhancements

// TODOc