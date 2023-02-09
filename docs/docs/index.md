# Welcome to Deep Yeast

*****

## Getting Started

### Installation

Clone Project
```
mkdir ${YOUR_PROJECT_NAME}
cd ${YOUR_PROJECT_NAME}
git clone https://github.com/wanlanli/deepYeast.git
```

Create conda environment
```
conda create -n deepyeast python=3.9`
conda activate deepyeast
conda install --file requirements.txt --channel=conda-forge --yes
```

Test your images
```
image = load_test_image()
model = load_trained_model()
output = model(image)
```
<fig>
</fig>

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        about.md  #
        ...       # Other markdown pages, images and other files.
