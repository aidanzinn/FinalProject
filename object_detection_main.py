from os.path import join
import sys

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from object_detection_helper import DetectionLitModule
from datasets.gen4_od_dataset import GEN1DetectionDataset

def collate_fn(batch):
    samples = [item[0] for item in batch]
    samples = torch.stack(samples, 0)
    
    targets = [item[1] for item in batch]
    return [samples, targets]

def main():
    class Config:
        # Dataset
        dataset = 'gen1'
        path = 'PropheseeGEN1'
        num_classes = 2

        # Data
        b = 64 # batch size
        sample_size = 100000
        T = 5 # Simulating time steps
        tbin = 2 # number of micro time bins
        image_shape = (240, 304)

        # Training
        epochs = 50
        lr = 1e-3 # learning rate
        wd = 1e-4 # weight decay
        num_workers = 4 # num of workers for dataloaders
        train = True  # Assuming default behavior is to train
        test = False  # Assuming default behavior is not to test
        device = 0
        precision = 16
        save_ckpt = False  # Assuming default behavior is not to save checkpoints
        comet_api = None  # Assuming no Comet API key by default

        # Backbone
        backbone = 'vgg-11' # NEED TO CHANGE TO CORRECT DENSENET
        use_batch_norm = True  # Assuming default behavior is to use BatchNorm2d
        pretrained_backbone = None
        pretrained = None
        extras = [640, 320, 320]

        # Priors
        min_ratio = 0.05
        max_ratio = 0.80
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        # Loss parameters
        box_coder_weights = [10.0, 10.0, 5.0, 5.0]
        iou_threshold = 0.50
        score_thresh = 0.01
        nms_thresh = 0.45
        topk_candidates = 200
        detections_per_image = 100

        # Replace argparse with the Config class
    args = Config()

    if args.dataset == "gen1":
        dataset = GEN1DetectionDataset #GEN1DetectionDataset # NEED TO REPLACE WITH GEN4 DATASET
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")

    module = DetectionLitModule(args) # THIS IS WHERE IT LINKS TO THE HELPER FILE
    
    # MIGHT NEED TO SET UP CHECKPOINTS HERE

    # Sets up "Trainer" object in pytorch lightning
    trainer = pl.Trainer(
        gpus=[args.device], gradient_clip_val=1., max_epochs=args.epochs,
        limit_train_batches=1., limit_val_batches=.25,
        check_val_every_n_epoch=5,
        deterministic=False,
        precision=args.precision,
    )
    if args.train:
        train_dataset = dataset(args, mode="train")
        val_dataset = dataset(args, mode="val")    
        train_dataloader = DataLoader(train_dataset, batch_size=args.b, collate_fn=collate_fn, num_workers=args.num_workers, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.b, collate_fn=collate_fn, num_workers=args.num_workers)
        
        trainer.fit(module, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()





