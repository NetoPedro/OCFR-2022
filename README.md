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
