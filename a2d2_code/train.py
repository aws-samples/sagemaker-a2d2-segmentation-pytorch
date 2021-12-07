# native
import argparse
import ast
import os
import time

# oss
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

# custom
from a2d2_utils import A2D2_S3_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # model & training parameters
    parser.add_argument("--epochs", type=int, default=1)
    # if used, iterations must be smaller than expected iterations in one epoch
    parser.add_argument("--iterations", type=int, default=10e5)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_warmup_ratio", type=float, default=1)
    parser.add_argument("--epoch_peak", type=int, default=2)
    parser.add_argument("--lr_decay_per_epoch", type=float, default=1)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--classes", type=int, default=55)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--eval-size", type=int, default=30)
    parser.add_argument("--height", type=int, default=1208)
    parser.add_argument("--width", type=int, default=1920)

    # infra configuration
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--amp", type=str, default="True")

    # Data, model, and output directories
    parser.add_argument("--cache", type=str, default="/tmp")
    parser.add_argument("--network", type=str, default="deeplabv3_mobilenet_v3_large")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--dataset", type=str, default=os.environ.get("SM_CHANNEL_DATASET"))
    parser.add_argument("--train-manifest", type=str, default="train_manifest.json")
    parser.add_argument("--val-manifest", type=str, default="val_manifest.json")
    parser.add_argument("--class-list", type=str, default="class_list.json")
    parser.add_argument("--bucket", type=str)

    args, _ = parser.parse_known_args()

    torch.cuda.empty_cache()

    # Instantiate data loader
    # ------------------------------------------------------------

    image_transform = Resize((args.height, args.width), interpolation=InterpolationMode.BILINEAR)

    target_transform = Resize((args.height, args.width), interpolation=InterpolationMode.NEAREST)

    # s3_resource = boto3.resource('s3')

    train_data = A2D2_S3_dataset(
        cache=args.cache,
        height=args.height,
        width=args.width,
        manifest_file=os.path.join(args.dataset, args.train_manifest),
        class_list=os.path.join(args.dataset, args.class_list),
        transform=image_transform,
        target_transform=target_transform,
        s3_bucket=args.bucket,
    )

    val_data = A2D2_S3_dataset(
        cache=args.cache,
        height=args.height,
        width=args.width,
        manifest_file=os.path.join(args.dataset, args.val_manifest),
        class_list=os.path.join(args.dataset, args.class_list),
        transform=image_transform,
        target_transform=target_transform,
        s3_bucket=args.bucket,
    )

    batch_size = args.batch

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch,
        persistent_workers=True,
    )

    # Instantiate model
    # ------------------------------------------------------------
    model = torch.hub.load(
        "pytorch/vision:v0.9.1", args.network, pretrained=False, num_classes=args.classes
    )

    model.train()

    amp = ast.literal_eval(args.amp)

    CE = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scaler = GradScaler(enabled=amp)

    # Training loop
    # ------------------------------------------------------------

    # Use gpu if available. This is a single-GPU script.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = args.epochs

    for epoch in range(0, num_epochs):

        # custom LR schedule. Warmup with peak and exponential decay
        if epoch <= args.epoch_peak:
            start_lr = args.lr * args.lr_warmup_ratio
            lr = start_lr + (epoch / args.epoch_peak) * (args.lr - start_lr)
        else:
            lr = args.lr * (args.lr_decay_per_epoch) ** (epoch - args.epoch_peak)

        print("In epoch {} learning rate: {:.10f}".format(epoch, lr))
        for p in optimizer.param_groups:
            p["lr"] = lr

        bstart = time.time()
        for i, batch in enumerate(train_loader):

            # if want to train for less than 1 epoch
            if i > args.iterations:
                break

            model.train()
            inputs = batch[0].to(device)
            masks = batch[1].to(device)
            optimizer.zero_grad()

            with autocast(enabled=amp):
                outputs = model(inputs)
                loss = CE(outputs["out"], masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i > 0 and (i / float(args.log_freq)).is_integer():

                bstop = time.time()
                val_losses = []
                model.eval()
                with torch.no_grad():
                    # validation dataloader takes 30s to load first batch :(...
                    for j, batch in enumerate(val_loader):
                        inputs = batch[0].to(device)
                        masks = batch[1].to(device)
                        outputs = model(inputs)
                        val_loss = CE(outputs["out"], masks)
                        val_losses.append(val_loss)
                        if j * batch_size >= args.eval_size:  # evaluate on a subset of val set
                            break
                avg_val_loss = torch.mean(torch.stack(val_losses))

                # print metrics
                throughput = float((i + 1) * batch_size) / (bstop - bstart)
                print("processed {} records in {}s".format(i * batch_size, bstop - bstart))
                print(
                    "batch {}: Training_loss: {:.4f}, Val_loss: {:.4f}, Throughput: {}".format(
                        i, loss, avg_val_loss, throughput
                    )
                )

                # save model twice ("latest" and versioned)
                checkpoint_name = "model-epoch{}-iter{}.pth".format(epoch, i)
                torch.save(model, os.path.join(args.checkpoint_dir, checkpoint_name))
                torch.save(model, os.path.join(args.checkpoint_dir, "latest_model.pth"))

    # we save the final model in the checkpoint location, for consistency
    torch.save(model, os.path.join(args.checkpoint_dir, "final_model.pth"))
