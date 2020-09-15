#!/usr/bin/env python3
"""Train VAE omic generator."""
import argparse
import json
import logging
import os
import pickle
import sys
import torch
from pytoda.datasets import GeneExpressionDataset
from paccmann_omics.decoders import DECODER_FACTORY
from paccmann_omics.encoders import ENCODER_FACTORY
from paccmann_omics.generators.vae import VAE
from paccmann_omics.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_omics.utils.utils import VAETracker, augment, get_device

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# yapf: disable
parser = argparse.ArgumentParser(description='Omics VAE training script.')
parser.add_argument(
    'train_filepath', type=str,
    help='Path to the training data (.csv).'
)
parser.add_argument(
    'val_filepath', type=str,
    help='Path to the validation data (.csv).'
)
parser.add_argument(
    'gene_filepath', type=str,
    help='Path to a pickle object containing list of genes.'
)
parser.add_argument(
    'model_path', type=str,
    help='Directory where the model will be stored.'
)
parser.add_argument(
    'params_filepath', type=str,
    help='Path to the parameter file.'
)
parser.add_argument(
    'training_name', type=str,
    help='Name for the training.'
)
# yapf: enable


def main(
    train_filepath, val_filepath, gene_filepath, model_path, params_filepath,
    training_name
):
    logger = logging.getLogger(f'{training_name}')
    logger.info(f'Params filename = {os.path.basename(params_filepath)}')
    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    # Create model directory and dump files
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp)

    # Prepare the dataset
    logger.info(' Start data preprocessing...')

    # Create dataset
    with open(gene_filepath, 'rb') as f:
        gene_list = pickle.load(f)

    train_dataset = GeneExpressionDataset(
        train_filepath, gene_list=gene_list, backend='eager'
    )
    train_mean, train_std = (
        train_dataset.processing['parameters']['mean'],
        train_dataset.processing['parameters']['std']
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=params['batch_size'], shuffle=True
    )
    val_dataset = GeneExpressionDataset(
        val_filepath,
        gene_list=gene_list,
        processing_parameters={
            'mean': train_mean,
            'std': train_std
        },
        backend='eager'
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=params['batch_size'], shuffle=True
    )

    t_s, v_s = next(
        iter(train_dataset)
    ).shape[0], next(iter(val_dataset)).shape[0]  # yapf: disable

    assert t_s == v_s, (
        f"Size discrepancies between training & validation,  ({t_s}, {v_s})."
    )
    assert params['input_size'] == t_s, (
        f"'input_size' in parameter json ({params['input_size']}) does not "
        f'match actual data shapes ({t_s}).'
    )

    device = get_device()
    save_top_model = os.path.join(model_dir, 'weights', '{}_{}_{}.pt')
    params.update({'save_top_model': save_top_model})

    # Define networks
    encoder = ENCODER_FACTORY[
        params.get('encoder_type', 'dense')
    ](params).to(device)    # yapf: disable
    decoder = DECODER_FACTORY[
        params.get('decoder_type', 'dense')
    ](params).to(device)    # yapf: disable
    model = VAE(params, encoder, decoder).to(device)

    logger.info(
        'Encoder type is {}'.format(
            str(type(encoder)).split('.')[-1].split("'")[0]
        )
    )
    # Define optimizer
    optimizer = (
        OPTIMIZER_FACTORY[params.get('optimizer', 'Adam')]
        (model.parameters(), lr=params.get('lr', 0.0005))
    )

    # Start training
    logger.info('Training about to start...')

    # Gradually decrease the alpha (weight of MSE relative to KL).
    alpha = params.get('alpha', 0.5)
    beta = params.get('beta', 1.)
    alphas = torch.cat(
        [
            torch.linspace(1, alpha, params.get('kl_annealing', 2)), alpha *
            torch.ones(params['epochs'] - params.get('kl_annealing', 2))
        ]
    ).double().to(device)

    epochs, latent_dim = params['epochs'], params['latent_size']
    tracker = VAETracker(
        logger, params, train_loader, val_loader, latent_dim, epochs
    )

    for epoch in range(epochs):

        tracker.new_train_epoch(epoch)
        model.train()
        logger.info(f"=== Epoch [{epoch}/{epochs}], -> Alpha {alphas[epoch]}")

        for ind, x in enumerate(train_loader):

            x_aug = augment(
                x,
                dropout=params.get('DAE_mask', 0.),
                sigma=params.get('DAE_noise', 0.)
            ).to(device)
            # Autoencoder part.
            x = x.to(device)
            x_fake = model(x_aug)  # encode to z and decode to x

            loss, rec, kld = model.joint_loss(x_fake, x, alphas[epoch], beta)
            optimizer.zero_grad()
            loss.backward()
            # Apply gradient clippping
            # torch.nn.utils.clip_grad_norm_(model.parameters(),1e-6)
            optimizer.step()
            tracker.update_train_batch(loss, rec, kld)

        tracker.logg_train_epoch()

        # Measure validation performance
        model.eval()
        with torch.no_grad():
            tracker.new_val_epoch(epoch)

            for x_val in val_loader:

                x_val = x_val.to(device)
                #  Reconstruction
                x_val_fake = model(x_val)
                loss, rec, kld = model.joint_loss(
                    x_val, x_val_fake, alpha, beta
                )
                tracker.update_val_batch(loss, rec, kld)

        tracker.logg_val_epoch()
        tracker.logg_tensorboard()
        tracker.check_to_save(encoder, decoder, model)

    tracker.final_log()
    tracker.save(encoder, decoder, model, 'training', 'done')
    logger.info("Done with training, models saved, shutting down.")


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the training
    main(
        args.train_filepath, args.val_filepath, args.gene_filepath,
        args.model_path, args.params_filepath, args.training_name
    )
