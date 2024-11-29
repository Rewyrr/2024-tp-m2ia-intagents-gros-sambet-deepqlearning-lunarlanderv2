
Ci-dessous:
* pour Mac et Linux, les commandes sont à faire dans un terminal classique. 
* Pour Windows, il faut utiliser **Anaconda prompt** et pas un terminal de commande classique (taper "Anaconda Prompt" dans la barre de recherche Windows). 

1. Créer (et activer) un nouvel environnement, appelé `tpdeeprl2023`:

```
conda create --name tpdeeprl2024 python=3.10
conda activate tpdeeprl2024
```

A ce niveau, votre ligne de commande doit ressembler à : `(tpdeeprl2024) <User>: `. 

`(tpdeeprl2024)` indique que l'environnement créé est actif, et vous pouvez maintenant installer des packages dans l'environnement.


2. Installation de PyTorch et torchvision:

-  Sur __Windows__: 
```
conda install pytorch==2.0.1  torchvision -c pytorch
conda install m2-base
```
- Sur __Mac__:
```
conda install pytorch==2.0.1  torchvision -c pytorch

```
- Sur __Linux__ : 
```
conda install pytorch=2.0.1 -c pytorch 
pip install torchvision
```
4. Git

Dans la suite il est supposé que `git` est installé sur votre machine. Si ce n'est pas le cas, vous pouvez utiliser `conda`pour l'installer:
```
conda install git
```

5. Cloner le dépôt créé *via* githubclassroom et aller dans le dossier du dépôt:
```
git clone https://github.com/X.git
cd X
```

6. Installation des packages spécifiés dans le fichier *requirements.txt*.
```
pip install -r requirements.txt
```
7. Installation de gymnasium
-  Sur __Windows__:
```
pip install swig
```
Ensuite aller sur https://visualstudio.microsoft.com/visual-cpp-build-tools/
 -> cliquer sur" Télécharger Build tools", puis lancer l'installer installé. Lors du choix, sélectionner "Desktop Development with C++"
Une fois installé:
```
pip install gymnasium[box2d]
```

- Sur __Linux__: 
```
pip install gymnasium
pip install gymnasium[box2d]
```
- Sur __Mac__:
```
conda install -c conda-forge gymnasium
conda install swig
conda install -c conda-forge gym-box2d
```


8. Vous pouvez maintenant  commencer à compléter le notebook `TPDQN.ipynb` pour faire votre TP, soit avec `jupyter-lab`, ou (conseillé) avec l'[extension Jupyter de VisualStudio](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) qui vous permet de debugger votre notebook.


## Sources

- si besoin, utiliser ([google colab](https://colab.research.google.com/?hl=fr)), version cloud de jupyter notebook qui  permet d'accéder gratuitement à des ressources informatiques, dont des GPU (limité).
- Un tutoriel sur les [Jupyter notebook](https://python.sdv.univ-paris-diderot.fr/18_jupyter/)
- Vous pouvez lister les environnements conda installés :
```
conda env list
```
- Vous pouvez lister les packages installés dans l'environnement actif :
```
conda list
```

