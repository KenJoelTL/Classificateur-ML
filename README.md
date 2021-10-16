# Classificateur de forme

Pour Démarrer le projet, copier l'ensemble A dans un nouveau répertoire nommé: images_dataset
Il n'est pas nécéssaire de choisir les types de Cercle/Diamant/Triangle/Hexagone. Le code s'occupera de créer un ensemble B seulement pour:

Cercle2
Cercle3
Diamant2
Diamant3
Hexagone2
Hexagone3
Triangle2
Triangle3

Une fois l'ensemble A copié dans images_dataset, exécuter la commande "python Main.py Generate"

À la fin de l'exécution vous aurez deux nouveaux fichiers "X.pkl" et "Y.pkl".

On peut maintenant exécuter les commandes suivantes :

1- Pour lancer le réseau de neuronne : "python Main.py [alpha][nombre d'itération] [nombre de noeud de la couche caché]"

2- Pour lancer un test de scalabilité : "python Main.py Scale [scale][alpha] [nombre d'itération][nombre de noeud de la couche caché]"

Les valeurs que nous avons utilisées sont: alpha: 0.0005 / nombre d'itération: 200 / nombre de noeud de la couche caché: 100

---

3- Pour lancer une comparaison avec un autre modèle (soit KNN, soit Naive Bayes, soit Regression Linéaire) : "python Main.py [model]" - Les valeurs possibles de model sont: KNN, Bayes ou Linear

Une fois l'exécution terminé les statistiques sont affichées directement dans la console.
