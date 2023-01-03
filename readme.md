
# DeepYeast

# DEMO

# Requirements
- Python3
- Numpy
- Tensorflow 2.10
- Scipy
- matplotlib
## Installation
### Git Clone Project
<code> 
mkdir ${YOUR_PROJECT_NAME}

cd ${YOUR_PROJECT_NAME}

git clone https://github.com/wanlanli/deepYeast.git

</code>

### Install Environment via conda
<code>
conda create -n deepyeast python=3.9

conda activate deepyeast

conda install --file requirements.txt --channel=conda-forge --yes 
</code>

## Running Test
### Dowload Weights and Data
smb://nasdcsr/RECHERCHE/FAC/FBM/DMF/smartin/cellfusion/LTS/2023_WL_NNmating/deepyeast_model/deepyeast_001
### Test Your Image
run example/01_DEMO to test your images