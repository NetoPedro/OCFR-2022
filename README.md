# OCFR-2022

Official repository for the **OCFR 2022: Competition on Occluded Face Recognition From Synthetically Generated Structure-Aware Occlusions** at 2022 International Joint Conference on Biometrics (IJCB 2022). 

The paper can be viewed at: [arXiv](https://arxiv.org/abs/2208.02760)



## Data

All the data used for the competition is available for download as a benchmark dataset. 

Images for the 8 different protocols are provided in the zip file, within individual folders. The list of images per protocol is provided in the form of **Reference_Probe_list.txt**, where Reference and Probe are replaced by the occlusion strategy **O[1-7]** or **C** (from Clear). 

The issame list is equal for all protocols, as well as the pairs list. Please be aware that the same image might appear non-occluded as a reference and occluded as a probe (while in different pairs). Hence, we provide a duplicate of all images with a renaming indicating if it is used as probe or not. 

We provide two images: the occlusion (a mask-like image where all the pixels except the occlusion pixels are set to 0), and an occlusion mask (similar to the previous, but occlusion pixels are set to 0 and the others are set to 255). To construct an occluded image, you should the respective [LFW](http://vis-www.cs.umass.edu/lfw/) image, multiply it by the occlusion mask and sum the occlusion. 

Download : [gdrive](https://drive.google.com/drive/folders/1ZtLYWvqbZW5NKcOq8nY5OIyS_i_xz820?usp=sharing)

## Generate new data

```
python3 align_db.py --input-dir=input_path --output-dir=output_path --image-size=112
```

## Replicating protocols

### 1
```
def select_occlusion_type():
    return np.random.choice([1,2,4]) 
```

### 2 
```
def select_occlusion_type():
    return np.random.choice([5,6,7,10,11]) 
```

### 3
```
def select_occlusion_type():
    return np.random.choice([1,2,4,5,6,7,10,11]) 
```
### 4 
```
def select_occlusion_type():
    return np.random.choice([1,2,4,6,7,10,11]) 
```
### 5
```
def select_occlusion_type():
    return np.random.choice([8,9]) 
```
### 6
```
def select_occlusion_type():
    return np.random.choice([5,6,7,8,9,10,11]) 
```
### 7 
```
def select_occlusion_type():
    return np.random.choice([1,2,4,5,6,7,8,9,10,11]) 
```
## Evaluation Script

Please also find a sample of an evaluation script in the **"evaluation.py"** file. 

## Including new occluders

We will release the code to generate the occluded images soon. Just by adding a **new** occluder image and its information in the **"occluders/occluders.csv"** you can use the script to create a new benchmark dataset. Hence, this dataset can be extended overtime with contributions from the community.  

Moreover, these occlusions can be easly applied to other **base** datasets. For instance, **AgeDB-30**, **MS1MV2**, etc. 

## Acknowledgement
The code was extended from the initial code of [Self-restrained-Triplet-Loss](https://github.com/fdbtrs/Self-restrained-Triplet-Loss). 

## Citation
If you use our code or data in your research, please cite with:

```
@article{neto2022ocfr,
  title={OCFR 2022: Competition on Occluded Face Recognition From Synthetically Generated Structure-Aware Occlusions},
  author = {Neto, Pedro C. and Boutros, Fadi and Pinto, Joao Ribeiro and Damer, Naser and Sequeira, Ana F. and Cardoso, Jaime S. and Bengherabi, Messaoud and Bousnat, Abderaouf and Boucheta, Sana and Hebbadj, Nesrine and Erakın, Mustafa Ekrem and Demir, Uğur and Ekenel, Hazım Kemal and Vidal, Pedro Beber de Queiroz and Menotti, David},
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

