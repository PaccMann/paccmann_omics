[![Build Status](https://github.com/PaccMann/paccmann_omics/actions/workflows/build.yml/badge.svg)](https://github.com/PaccMann/paccmann_omics/actions/workflows/build.yml)
# paccmann_omics

Generative models of omic data for PaccMann<sup>RL</sup>.

`paccmann_omics` is a package to model omic data, with examples for generative 
models of gene expression profiles and encoded proteins (vector representations).
For example, see our papers:
- [_PaccMann<sup>RL</sup>: De novo generation of hit-like anticancer molecules from transcriptomic data via reinforcement learning_](https://www.cell.com/iscience/fulltext/S2589-0042(21)00237-6) (_iScience_, 2021). In there, we use a denoising, dense VAE to model gene expression profiles from TCGA (code in this repo). We then use these encodings to conditionally generate de novo molecules with high predicted efficacy against these cell types.
- [Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2](https://iopscience.iop.org/article/10.1088/2632-2153/abe808) (_Machine Learning: Science and Technology_, 2021). In there, we use a denoising, dense VAE to model proteins from UniProt (code in this repo). We then use a set of 41 SARS-CoV-2 related proteins to conditionally generate de novo molecules with high predicted binding affinity against these proteins.

## Requirements

- `conda>=3.7`

## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements. 
To run the example training script we provide environment files under `examples/`.

Create a conda environment:

```sh
conda env create -f examples/gene_expression/conda.yml
```

Activate the environment:

```sh
conda activate paccmann_omics
```

Install in editable mode for development:

```sh
pip install -e .
```

## Example usage

In the `examples` directory is a training script `train_vae.py` that makes use
of paccmann_omics.

```console
(paccmann_omics) $ python examples/gene_expression/train_vae.py -h
usage: train_vae.py [-h]
                    train_filepath val_filepath gene_filepath model_path
                    params_filepath training_name

Omics VAE training script.

positional arguments:
  train_filepath   Path to the training data (.csv).
  val_filepath     Path to the validation data (.csv).
  gene_filepath    Path to a pickle object containing list of genes.
  model_path       Directory where the model will be stored.
  params_filepath  Path to the parameter file.
  training_name    Name for the training.

optional arguments:
  -h, --help       show this help message and exit
```

`params_filepath` could point to [examples/gene_expression/example_params.json](examples/gene_expression/example_params.json), examples for other files can be downloaded from [here](https://ibm.box.com/v/paccmann-pytoda-data).

## References

If you use `paccmann_omics` in your projects, please cite the following:

```bib
@article{born2021datadriven,
  author = {Born, Jannis and Manica, Matteo and Cadow, Joris and Markert, Greta and Mill, Nil Adell and Filipavicius, Modestas and Janakarajan, Nikita and Cardinale, Antonio and Laino, Teodoro and {Rodr{\'{i}}guez Mart{\'{i}}nez}, Mar{\'{i}}a},
  doi = {10.1088/2632-2153/abe808},
  issn = {2632-2153},
  journal = {Machine Learning: Science and Technology},
  number = {2},
  pages = {025024},
  title = {{Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2}},
  url = {https://iopscience.iop.org/article/10.1088/2632-2153/abe808},
  volume = {2},
  year = {2021}
}

@article{born2021paccmannrl,
  title = {PaccMann\textsuperscript{RL}: De novo generation of hit-like anticancer molecules from transcriptomic data via reinforcement learning},
  journal = {iScience},
  volume = {24},
  number = {4},
  pages = {102269},
  year = {2021},
  issn = {2589-0042},
  doi = {https://doi.org/10.1016/j.isci.2021.102269},
  url = {https://www.cell.com/iscience/fulltext/S2589-0042(21)00237-6},
  author = {Born, Jannis and Manica, Matteo and Oskooei, Ali and Cadow, Joris and Markert, Greta and {Rodr{\'{i}}guez Mart{\'{i}}nez}, Mar{\'{i}}a}
}


```
