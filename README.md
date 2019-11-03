[![Build Status](https://travis-ci.org/PaccMann/paccmann_omics.svg?branch=master)](https://travis-ci.org/PaccMann/paccmann_omics)
# paccmann_omics

Generative models of omic data for PaccMann^RL.

`paccmann_omics` is a package to model omic data, currently generative models  
of gene expression profiles.

## Requirements

- `conda>=3.7`

## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements. 
To run the example training script we provide environment files under `examples/`.

Create a conda environment:

```sh
conda env create -f examples/conda.yml
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
(paccmann_omics) $ python examples/train_vae.py -h
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

`params_filepath` could point to [examples/example_params.json](examples/example_params.json), examples for other files can be downloaded from [here](https://ibm.box.com/v/paccmann-pytoda-data).

## References

If you use `paccmann_omics` in your projects, please cite the following:

```bib
@misc{born2019reinforcement,
    title={Reinforcement learning-driven de-novo design of anticancer compounds conditioned on biomolecular profiles},
    author={Jannis Born and Matteo Manica and Ali Oskooei and Maria Rodriguez Martinez},
    year={2019},
    eprint={1909.05114},
    archivePrefix={arXiv},
    primaryClass={q-bio.BM}
}
```
