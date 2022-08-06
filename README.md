# OCFR-2022

Official repository for the **OCFR 2022: Competition on Occluded Face Recognition From Synthetically Generated Structure-Aware Occlusions** at 2022 International Joint Conference on Biometrics (IJCB 2022). 

The paper can be viewed at: [arXiv](https://arxiv.org/abs/2208.02760)


## Evaluation Script

Please also find a sample of an evaluation script in the **"evaluation.py"** file. 


## Data

All the data used for the competition is available for download as a benchmark dataset. 

Images for the 8 different protocols are provided in the zip file, within individual folders. The list of images per protocol is provided in the form of **Reference_Probe_list.txt**, where Reference and Probe are replaced by the occlusion strategy **O[1-7]** or **C** (from Clear). 

The issame list is equal for all protocols, as well as the pairs list. Please be aware that the same image might appear non-occluded as a reference and occluded as a probe (while in different pairs). Hence, we provide a duplicate of all images with a renaming indicating if it is used as probe or not. 

Download : [gdrive](https://drive.google.com/drive/folders/1ZtLYWvqbZW5NKcOq8nY5OIyS_i_xz820?usp=sharing)


## Including new occluders

We will release the code to generate the occluded images soon. Just by adding a **new** occluder image and its information in the **"occluders.csv"** you can use the script to create a new benchmark dataset. Hence, this dataset can be extended overtime with contributions from the community.  

Moreover, these occlusions can be easly applied to other **base** datasets. For instance, **AgeDB-30**, **MS1MV2**, etc. 

## Acknowledgement
The code was extended from the initial code from [Self-restrained-Triplet-Loss](https://github.com/fdbtrs/Self-restrained-Triplet-Loss). 

## Citation
If you use our code or data in your research, please cite with:

@article{neto2022ocfr,
  title={OCFR 2022: Competition on Occluded Face Recognition From Synthetically Generated Structure-Aware Occlusions},
  author = {Neto, Pedro C. and Boutros, Fadi and Pinto, Joao Ribeiro and Damer, Naser and Sequeira, Ana F. and Cardoso, Jaime S. and Bengherabi, Messaoud and Bousnat, Abderaouf and Boucheta, Sana and Hebbadj, Nesrine and Yahya-Zoubir, Bahia and Erakın, Mustafa Ekrem and Demir, Uğur and Ekenel, Hazım Kemal and Vidal, Pedro Beber de Queiroz and Menotti, David},
  journal={arXiv preprint arXiv:2208.02760},
  year={2022}
}


## TODO 

- [X] Link to the data
- [X] Link to the preprint
- [X] Citation to arXiv
- [ ] Link to the published paper
- [ ] Citation published version
- [ ] Evaluation script
- [ ] Upload occluders
- [ ] Script to generate the data
