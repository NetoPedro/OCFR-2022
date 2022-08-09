# OCFR-2022

Official repository for the **OCFR 2022: Competition on Occluded Face Recognition From Synthetically Generated Structure-Aware Occlusions** at 2022 International Joint Conference on Biometrics (IJCB 2022). 

The paper can be viewed at: [arXiv](https://arxiv.org/abs/2208.02760)


## Evaluation Script

Please also find a sample of an evaluation script in the **"evaluation.py"** file. 

## Including new occluders

We will release the code to generate the occluded images soon. Just by adding a **new** occluder image and its information in the **"occluders.csv"** you can use the script to create a new benchmark dataset. Hence, this dataset can be extended overtime with contributions from the community.  

Moreover, these occlusions can be easly applied to other **base** datasets. For instance, **AgeDB-30**, **MS1MV2**, etc. 

## Acknowledgement
The code was extended from the initial code of [Self-restrained-Triplet-Loss](https://github.com/fdbtrs/Self-restrained-Triplet-Loss). 

## Citation
If you use our code or data in your research, please cite with:

```
@article{neto2022ocfr,
  title={OCFR 2022: Competition on Occluded Face Recognition From Synthetically Generated Structure-Aware Occlusions},
  author = {Neto, Pedro C. and Boutros, Fadi and Pinto, Joao Ribeiro and Damer, Naser and Sequeira, Ana F. and Cardoso, Jaime S. and Bengherabi, Messaoud and Bousnat, Abderaouf and Boucheta, Sana and Hebbadj, Nesrine and Yahya-Zoubir, Bahia and Erakın, Mustafa Ekrem and Demir, Uğur and Ekenel, Hazım Kemal and Vidal, Pedro Beber de Queiroz and Menotti, David},
  journal={arXiv preprint arXiv:2208.02760},
  year={2022}
}
```

and 

```
@TechReport{LFWTech,
  author =       {Gary B. Huang and Manu Ramesh and Tamara Berg and 
                  Erik Learned-Miller},
  title =        {Labeled Faces in the Wild: A Database for Studying 
                  Face Recognition in Unconstrained Environments},
  institution =  {University of Massachusetts, Amherst},
  year =         2007,
  number =       {07-49},
  month =        {October}}
```

## TODO 

- [X] Link to the preprint
- [X] Citation to arXiv
- [ ] Link to the published paper
- [ ] Citation published version
- [ ] Evaluation script
- [ ] Upload occluders
- [ ] Script to generate the data

## License
```
Attribution-NonCommercial-ShareAlike 4.0 International
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0) license. 
Copyright (c) 2022 Instituto de Engenharia de Sistemas e Computadores, Tecnologia e Ciência
```

