# Data Science × Régression logistique

## Harry Potter et le Data Scientist

*Résumé : Écrivez un classifieur et sauvez Poudlard !*

*Version : 5*

---

## Table des matières

- [I. Avant-propos](#chapitre-i--avant-propos)
- [II. Introduction](#chapitre-ii--introduction)
- [III. Objectifs](#chapitre-iii--objectifs)
- [IV. Consignes générales](#chapitre-iv--consignes-générales)
- [V. Partie obligatoire](#chapitre-v--partie-obligatoire)
  - [V.1 Analyse des données](#v1-analyse-des-données)
  - [V.2 Visualisation des données](#v2-visualisation-des-données)
    - [V.2.1 Histogramme](#v21-histogramme)
    - [V.2.2 Nuage de points](#v22-nuage-de-points)
    - [V.2.3 Pair plot](#v23-pair-plot)
  - [V.3 Régression logistique](#v3-régression-logistique)
- [VI. Partie bonus](#chapitre-vi--partie-bonus)
- [VII. Rendu et évaluation par les pairs](#chapitre-vii--rendu-et-évaluation-par-les-pairs)
- [VIII. Annexes](#chapitre-viii--annexes)
  - [VIII.1 Mathématiques](#viii1-mathématiques)
  - [VIII.2 Exemples de visualisation de données](#viii2-exemples-de-visualisation-de-données)
---

## Chapitre I : Avant-propos

Voici ce que Wikipédia dit de Yann Le Cun, l’un des pères fondateurs de l’IA :

> Yann Le Cun est né près de Paris, en France, en 1960. Il est chercheur en intelligence artificielle et en vision par ordinateur (robotique). Il est considéré comme l’un des inventeurs de l’apprentissage profond.
>
> Diplômé de l’ESIEE Paris en 1983, il obtient ensuite un doctorat à l’Université Pierre-et-Marie-Curie en 1987, au cours duquel il propose une forme primitive de l’algorithme d’apprentissage par rétropropagation, couramment utilisé par les algorithmes d’optimisation par descente de gradient pour ajuster le poids des neurones en calculant le gradient de la fonction de coût. Il est chercheur postdoctoral dans le laboratoire de Geoffrey Hinton à l’Université de Toronto de 1987 à 1988.
>
> Depuis les années 1980, Yann Le Cun travaille sur l’apprentissage automatique, la vision par ordinateur et l’apprentissage profond : la capacité d’un ordinateur à reconnaître des représentations (images, textes, vidéos, sons) en l’exposant de manière répétée à des exemples d’entraînement.
>
> En 1987, Yann Le Cun rejoint l’Université de Toronto, puis en 1988 les laboratoires de recherche d’AT&T, où il développe des méthodes d’apprentissage supervisé.
>
> Yann Le Cun est également l’un des principaux créateurs de la technologie de compression d’images DjVu (avec Léon Bottou et Patrick Haffner).
>
> Yann Le Cun est professeur à l’Université de New York, où il a créé le Center for Data Science. Il travaille notamment au développement technologique des voitures autonomes.
>
> Le 9 décembre 2013, Yann Le Cun est invité par Mark Zuckerberg à rejoindre Facebook pour concevoir et diriger le laboratoire d’intelligence artificielle FAIR (« Facebook Artificial Intelligence Research ») à New York, Menlo Park et, depuis 2015, à Paris, afin de travailler sur la reconnaissance d’images. Il avait auparavant refusé une proposition similaire de Google.
>
> En 2016, il est professeur invité d’informatique à la « Chaire annuelle Informatique et Sciences numériques » du Collège de France à Paris.

---

## Chapitre II : Introduction

Oh non ! Depuis sa création, la célèbre école de sorcellerie Poudlard n’avait jamais connu un tel affront. Les forces du mal ont ensorcelé le Choixpeau magique. Il ne répond plus et est incapable de remplir son rôle : répartir les élèves dans les maisons.

La rentrée approche. Heureusement, le professeur McGonagall a su agir dans une situation aussi stressante, car il est impossible que Poudlard n’accueille pas de nouveaux élèves… Elle a décidé de faire appel à vous, un « datascientist » moldu capable de faire des miracles avec l’outil que tous les moldus savent utiliser : un « ordinateur ».

Malgré la réticence naturelle de nombreux sorciers, la directrice de l’école vous reçoit dans son bureau pour vous expliquer la situation. Vous êtes ici parce que son informateur a découvert que vous êtes capable de recréer un Choixpeau magique à l’aide de vos outils de moldu. Vous lui expliquez que, pour que vos outils « moldus » fonctionnent, vous avez besoin des données des élèves. Avec hésitation, le professeur McGonagall vous remet un grimoire poussiéreux. Heureusement pour vous, un simple « Digitalis ! » et le livre se transforme en clé USB.

---

## Chapitre III : Objectifs

Dans ce projet Data Science × Régression logistique, vous allez poursuivre votre exploration du *Machine Learning* en découvrant différents outils.

L’utilisation du terme *Data Science* dans le titre peut être jugée excessive par certains. C’est vrai. Nous ne prétendons pas vous donner toutes les bases de la *Data Science* dans ce sujet. Le domaine est vaste. Nous n’aborderons que quelques bases qui nous semblent utiles pour l’exploration des données avant de les transmettre à l’algorithme de *machine learning*.

Vous allez implémenter un modèle de classification linéaire dans la continuité du sujet sur la *régression linéaire* : une *régression logistique*. Nous vous encourageons également à vous constituer une boîte à outils de *machine learning* au fil de votre progression.

En résumé :

- Vous apprendrez à lire un jeu de données, à le visualiser de différentes manières, et à sélectionner et nettoyer les informations inutiles de vos données.
- Vous entraînerez un modèle de régression logistique qui résoudra un problème de classification.
---

## Chapitre IV : Consignes générales

Vous pouvez utiliser le langage de votre choix. Nous vous recommandons toutefois de choisir un langage disposant d’une bibliothèque facilitant le tracé de graphiques et le calcul des propriétés statistiques d’un jeu de données.

Toute méthode qui fait tout le gros du travail à votre place (par exemple, la méthode `describe()` d’un DataFrame `pandas`) sera considérée comme de la triche.

---

## Chapitre V : Partie obligatoire

> ⚠️ **Il est vivement conseillé de suivre les étapes dans cet ordre.**

### V.1 Analyse des données

> ℹ️ Nous verrons quelques étapes de base de l’exploration de données. Ce ne sont évidemment pas les seules techniques ni les seules étapes possibles. Chaque jeu de données et chaque problème s’abordent de façon unique. Vous trouverez sûrement d’autres façons d’analyser vos données.

Commencez par examiner les données disponibles : leur format, la présence de différents types de données, leurs plages de valeurs, etc. Il est important de bien connaître sa matière première avant de commencer. Plus vous travaillerez avec des données, plus vous développerez une intuition sur la manière de les exploiter.

Dans cette partie, le professeur McGonagall vous demande de produire un programme appelé `describe.[extension]`. Il prend un *dataset* en paramètre et doit simplement afficher les informations de toutes les *features* numériques, comme ici :

```
$> describe.[extension] dataset_train.csv
             Feature 1     Feature 2     Feature 3     Feature 4
Count       149.000000    149.000000    149.000000    149.000000
Mean          5.848322      3.051007      3.774497      1.205369
Std           5.906338      3.081445      4.162021      1.424286
Min           4.300000      2.000000      1.000000      0.100000
25%           5.100000      2.800000      1.600000      0.300000
50%           5.800000      3.000000      4.400000      1.300000
75%           6.400000      3.300000      5.100000      1.800000
Max           7.900000      4.400000      6.900000      2.500000
```

> ⚠️ **Il est interdit d’utiliser toute fonction qui fait le travail à votre place, comme : count, mean, std, min, max, percentile, etc., quel que soit le langage utilisé. Il est bien sûr également interdit d’utiliser la bibliothèque describe ou toute fonction (plus ou moins) similaire d’une autre bibliothèque.**

### V.2 Visualisation des données

La visualisation des données est un outil puissant pour un data scientist. Elle permet d’en tirer des enseignements et de développer une intuition sur l’allure de vos données. Visualiser vos données permet aussi de détecter des défauts ou des anomalies.

Dans cette section, il vous est demandé de créer un ensemble de scripts, chacun utilisant une méthode de visualisation particulière pour répondre à une question. Il n’y a pas forcément une seule réponse à la question.

#### V.2.1 Histogramme

Créez un script appelé `histogram.[extension]` qui affiche un *histogramme* répondant à la question suivante :

> Quel cours de Poudlard a une répartition des notes homogène entre les quatre maisons ?

#### V.2.2 Nuage de points

Créez un script appelé `scatter_plot.[extension]` qui affiche un *nuage de points* (scatter plot) répondant à la question suivante :

> Quelles sont les deux features qui se ressemblent ?

#### V.2.3 Pair plot

Créez un script appelé `pair_plot.[extension]` qui affiche un *pair plot* ou une *matrice de nuages de points* (selon la bibliothèque que vous utilisez).

À partir de cette visualisation, quelles features allez-vous utiliser pour votre régression logistique ?

### V.3 Régression logistique

Vous arrivez à la dernière partie : codez votre Choixpeau magique. Pour cela, vous devez réaliser un multi-classifieur par régression logistique *one-vs-all* (*one-vs-rest*).

Vous devrez créer deux programmes :

- Le premier, nommé `logreg_train.[extension]`, entraînera vos modèles. Il prend `dataset_train.csv` en paramètre. Pour la partie obligatoire, vous devez minimiser l’erreur par *descente de gradient*. Le programme génère un fichier contenant les poids utilisés pour la prédiction.
- Le second doit s’appeler `logreg_predict.[extension]`. Il prend en paramètres `dataset_test.csv` et un fichier contenant les poids entraînés par le programme précédent.
  Pour évaluer votre classifieur, ce second programme devra générer un fichier de prédiction `houses.csv` formaté exactement ainsi :
```
$> cat houses.csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
3,Hufflepuff
4,Slytherin
5,Ravenclaw
6,Hufflepuff
[...]
```

---

## Chapitre VI : Partie bonus

Il existe de nombreux bonus intéressants à ajouter à ce sujet. Voici quelques suggestions :

- Ajouter des champs supplémentaires à `describe.[extension]`
- Implémenter une *descente de gradient stochastique*
- Implémenter d’autres algorithmes d’optimisation (Batch GD, mini-batch GD, ou autres)
> ⚠️ **La partie bonus ne sera évaluée que si la partie obligatoire est PARFAITE. Parfaite signifie que la partie obligatoire a été entièrement réalisée et fonctionne sans aucun problème. Si vous n’avez pas rempli TOUTES les exigences obligatoires, votre partie bonus ne sera pas évaluée.**

---

## Chapitre VII : Rendu et évaluation par les pairs

Rendez votre travail sur votre dépôt `Git` comme d’habitude. Seul le travail présent dans votre dépôt sera évalué lors de la soutenance. N’hésitez pas à vérifier deux fois les noms de vos dossiers et fichiers pour vous assurer qu’ils sont corrects.

Lors de la correction, vous serez évalué sur votre rendu (aucune fonction qui fait tout le gros du travail à votre place), ainsi que sur votre capacité à présenter, expliquer et justifier vos choix.

Votre classifieur sera évalué sur les données présentes dans `dataset_test.csv`. Vos réponses seront évaluées à l’aide de l’[accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) de la bibliothèque *Scikit-Learn*. Le professeur McGonagall considère que votre algorithme est comparable au Choixpeau magique uniquement s’il obtient un accuracy score minimum de **98 %**.

Il sera également important de savoir expliquer le fonctionnement des algorithmes de *machine learning* utilisés.

---

## Chapitre VIII : Annexes

### VIII.1 Mathématiques

La régression logistique fonctionne presque comme la régression linéaire. Voici une fonction de coût (*loss*) :

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{i} \log\left(h_\theta(x^{i})\right) + (1 - y^{i}) \log\left(1 - h_\theta(x^{i})\right) \right]
$$

Où $h_\theta(x)$ est défini de la façon suivante :

$$
h_\theta(x) = g(\theta^{T} x)
$$

Avec :

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

La fonction de coût nous donne la dérivée partielle suivante :

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{i}) - y^{i} \right) x_j^{i}
$$

### VIII.2 Exemples de visualisation de données

Voici quelques exemples de visualisation de données :

- **Histogramme** : *(image non reprise dans ce fichier : histogramme superposé de 3 features)*
- **Nuage de points** : *(image non reprise dans ce fichier : nuage de points de 2 classes selon Feature 1 et Feature 2)*
- **Pair plot** : *(image non reprise dans ce fichier : matrice de nuages de points de 4 features, 3 classes)*
 