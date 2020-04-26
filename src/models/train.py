import json
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split

from ..features.datasets import AudioDataset
from .vae import VAE
from ..data.fs_utils import git_root
from .logger import get_logger
from ..visualization.save import save_wav


def train_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    device,
    logger
):
    model.train()
    train_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_fn(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

    logger.info('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test_epoch(
    epoch,
    model,
    test_loader,
    device,
    out_name,
    sampling_rate,
    logger
):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            # export first batch for comparison
            if i == 0:
                dshape = data.size()

                # insert half second silence between sample and its reconstruction
                silence = torch.zeros(dshape[0], sampling_rate // 2)
                comparison = torch.cat(
                    [data, silence, recon_batch],
                    dim=1
                ).cpu()

                for j in range(dshape[0]):
                    save_wav(
                        out_name,
                        f'epoch{str(epoch)}_comparison_{str(j)}',
                        comparison[j, :],
                        sampling_rate
                    )

    test_loss /= len(test_loader.dataset)
    logger.info('====> Test set loss: {:.4f}'.format(test_loss))


def main(
    dataset_dir,
    seed,
    cuda,
    sample_length,
    sampling_rate,
    batch_size,
    test_split,
    epochs,
    output_prefix,
    log_level,
    **kwargs
):
    logger = get_logger('train', log_level)
    run_timestamp = datetime.now().strftime('%H-%M-%S')
    out_name = f'{output_prefix}_{run_timestamp}'

    logger.info('========================================')
    logger.info('')
    logger.info(f'======= starting run {out_name} ========')

    torch.manual_seed(seed)
    logger.debug(f'seed set to {seed}')

    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    loader_args = {
        'num_workers': 1,
        'pin_memory': True
    } if cuda else {}

    logger.debug(f'cuda set to {cuda}')

    full_ds = AudioDataset(
        f'{git_root()}/{dataset_dir}',
        maxlen=sample_length,
        sampling_rate=sampling_rate
    )

    train_ds, test_ds = random_split(
        full_ds,
        [len(full_ds) - int(len(full_ds) * test_split), int(len(full_ds) * test_split)]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_args
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_args
    )

    logger.debug(f'loading dataset from {dataset_dir} ({test_split} test split) with')
    logger.debug(f'sample length {sample_length} seconds')
    logger.debug(f'sampling rate {sampling_rate} Hz')
    logger.debug(f'batch size {batch_size}')

    model = VAE(
        input_size=sample_length * sampling_rate
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # TODO: extract lr?

    logger.info(f'starting training for {epochs} epochs')
    for epoch in range(1, epochs + 1):
        train_epoch(epoch, model, train_loader, optimizer, device, logger)
        test_epoch(epoch, model, test_loader, device, out_name, sampling_rate, logger)

    logger.info(f'training finished, saving model under {output_prefix} prefix')
    save_model(model, 'models', out_name)

    logger.info('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='path to training configuration json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    main(**cfg)
