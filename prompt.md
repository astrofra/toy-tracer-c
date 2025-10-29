# Toy tracer prompt to ChatGPT, 2025-10-29

Je réflechis à un projet de cours pour une classe de Master M1 qui débutent en programmation C. Ils ont des PC portables simples sous Windows.
Je voudrais les amener à programmer en C, aussi facilement qu'on peut le faire avec VSCode. Il faudrait qu'ils aient un minimum de fichiers à gérer, aucune dépendance, pas de cmake ni de make.

Ce pourrait etre : 
- main.c // Main structure, mostly how the renderer can be invoked
- tracer.c // Main raytracing implementation
- tracer.h // Main raytracing API
- scene.c // User description of the scene (the function already exist but it is mostly empty, only a cube on a checkerboard ?)
- sceneH // User headers, defines, ...

Le raytraceur ne pourrait rendre que les objets primitives de types suivants :
- plan
- cube
- sphere
- cylindre
- torus

Seule la pointlight serait supportée, avec ou sans ombres portées (en raytrace)
Pour chaque primitive déclarée dans la scene, on pourrait invoquer un struct qui serait un material, permettant de définir : couleur (albedo), roughness, metalness
Le "material" serait traité par une fonction en C qui serait une implémentation simple mais efficace du "principled PBR" de Blender par exemple
On ne pourrais pas gérer les textures

L'idée est de proposer à chaque étudiant de modifier, compiler, débugger, lancer son implementation de scene en "plaçant" ses objets en code dans scene.c

L'image produite serait dumpée dans un format très simple, par exemple du TGA32
On pourrait facilement compiler sur une machine Windows, avec un compilateur léger (MinGW ?) qui pourrait etre débuggé visuellement sous Visual Studio
Sur Mac, ce serait le compilateur par défaut de la machine
Sous Linux, pareil

Le code serait totalement portable, avec un minumum de spécificté à la machine. Le raytraceur n'afficherait aucune image, juste il cracherait des fichier TGA :)

J'ai donc besoin que tu rédiges un AGENTS.md qui détaille précisement le cahier des charges de ce projet afin que CODEX puisse en écrire le code en C.

- pas C++
- pas de make
- pas de cmake
- je peux pouvoir télécharger le markdown 
- everything shall be in english
- code comments in english
