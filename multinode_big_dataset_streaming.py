
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os
from mpi4py import MPI
import socket
from torchvision import datasets, transforms
import math

from datasets import load_from_disk
from base_transformer import TransformerModel
from mixer_model import KronMixer
from datasets.distributed import split_dataset_by_node
from huggingface_dset_preprocess import tokenize
from datasets import load_dataset
import pickle

def ddp_setup(config):
    backend = 'nccl'
    #local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    local_rank = rank%config.ngpus_per_node
    print('World size:', size)
    print('Rank: ', rank)
    print('local rank:', local_rank)
    # Pytorch will look for these:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None
    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        tokenizer,
        config,
        dset,
        wandb_run = None,
        grad_clip = 0,
        grad_accumulate = 1,
    ) -> None:
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        print('global rank: ', self.global_rank)
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.config = config
        self.epochs_run = 0
        self.train_steps = 0
        self.dset = dset
        self.grad_clip = grad_clip
        self.grad_accumulate = grad_accumulate
        self.train_losses = []
        self.downcasted = False
        self.save_path = '../../../lus/eagle/projects/tpc/sww/models/' + config.wandb_name + '.pt'
        if os.path.exists(self.save_path):
            print("Loading snapshot")
            self._load_snapshot()
        self.model = DDP(self.model, device_ids=[self.local_rank])
        if config.use_wandb and int(os.environ["RANK"])==0:
            self.wandb_run = wandb_run
            self.wandb_run.watch(self.model.module)
            self.wandb_run.config.num_params = sum(p.numel() for p in self.model.module.parameters() if p.requires_grad)
        self.last_loss=0
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def _load_snapshot(self):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(self.save_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.train_steps = snapshot["train_steps"]
        print(f"Resuming training from snapshot at {self.train_steps} training steps")
    
    def _svd_prune(self, prune_level):
        with torch.no_grad():
            self.model.module = self.model.module.float()
            layers = list(self.model.module.transformer_encoder.layers)
            layers.append(self.model.module.linear)
            for layer in layers:
                for name, param in layer.named_parameters():
                    if len(param.shape) > 1:
                        u, s, v = torch.svd(param)
                        s[s<prune_level] = 0
                        param.copy_(u @ torch.diag(s) @ v.t())
            self.model.module = self.model.module.half()

    def _run_batch(self, source, targets, idx):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output.transpose(1,2), targets)/self.grad_accumulate
        self.last_loss += loss
        loss.backward()
        if (idx + 1) % self.grad_accumulate == 0:
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.train_steps += 1
            return self.last_loss
        else:
            return None

    def _run_epoch(self, epoch):
        print('RUNNING EPOCH')
        # self.train_data.sampler.set_epoch(epoch)
        self.dset.set_epoch(epoch)
        train_loss = 0
        min_loss = float('inf')
        losses = ''
        for idx, source in enumerate(self.train_data):
            source = source['input_ids']
            targets = source[:, 1:]
            source = source[:, :-1]
            if idx > self.config.prune_warmup:
                if self.downcasted is False and self.config.downcast:
                    self.model.module = self.model.module.half()
                    self.downcasted = True
                if idx % self.config.prune_level ==0:
                    dist.barrier()
                    self._svd_prune(self.config.prune_level)
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            train_loss = self._run_batch(source, targets, idx)
            if train_loss is not None:
                self.last_loss = 0
                if self.global_rank == 0:
                    # print(f'train loss: {train_loss}')
                    losses += f'step: {idx}, loss: {train_loss}\n'
            # eval code from: https://discuss.pytorch.org/t/ddp-evaluation-tensorboard-logging/175480
                if (idx + 1) % self.config.eval_every == 0 and self.global_rank == 0:
                    with open(f'{self.config.wandb_name}_loss_log.txt','a') as f:
                        f.write(losses)
                        losses = ''
                    # print('wandb logging step')
                    self.wandb_run.log({f'train loss every {self.config.eval_every} steps':train_loss})
                    self.wandb_run.log({'total train steps':idx + 1})
                    if train_loss < min_loss and  idx % config.save_every == 0:
                        min_loss = train_loss
                        self._save_snapshot()
    # eval code from: https://discuss.pytorch.org/t/ddp-evaluation-tensorboard-logging/175480     
    def _validate_batch(self, source, targets):
        with torch.no_grad():
            output = self.model(source)
            return F.cross_entropy(output.transpose(1,2), targets).item()

    def _get_val_loss(self):
        val_loss = 0
        for source in self.val_data:
            source = source['input_ids']
            targets = source[:, 1:]
            source = source[:, :-1]
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            val_loss += self._validate_batch(source, targets)
        return val_loss / len(self.val_data)

    def _save_snapshot(self):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
        }
        torch.save(snapshot, self.save_path)

    def train(self):
        for epoch in range(self.epochs_run, self.config.total_epochs):
            self._run_epoch(epoch)
            self.epochs_run += 1
            dist.barrier()
            # if self.global_rank == 0 and self.config.use_wandb:
            #     self.wandb_run.log({f'val loss by epoch':self._get_val_loss()})

def warmup_cosine_decay(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / warmup_steps
    elif step > warmup_steps and step < total_steps:
        # from 1 to 0.1
        return 0.45 * (math.cos(math.pi * (step - warmup_steps) / total_steps) + 1.0) + 0.1
    else:
        return 0.1
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_decay)

def load_train_objs(config):
    if config.arch == 'kron':
        model = KronMixer(max_len = config.train_seq_len, d_model = config.d_model, hidden_dim = config.d_hid, 
        vocab_size = config.vocab_size, depth = config.n_layers, heads = config.n_heads, dropout = config.dropout, kron_type = config.kron_block_type, num_proxies = config.num_proxies)
    elif config.arch == 'tfmr':
        model = TransformerModel(ntoken = config.vocab_size, d_model = config.d_model, nhead = config.n_heads, d_hid = config.d_hid, 
        nlayers = config.n_layers, dropout = config.dropout)
    if config.use_galore:
        from galore_torch import GaLoreAdamW
        weights = []
        biases = []
        for param in model.parameters():
            if len(param.shape) > 1:
                weights.append(param)
            else:
                biases.append(param)

        param_groups = [{'params':biases},
            {'params':weights, 'rank':128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}]
        optimizer = GaLoreAdamW(param_groups, lr = config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.95))
    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: warmup_cosine_decay(step, config.warmup_steps, config.total_steps))
    else:
        scheduler = None
    tokenizer = None
    return model, optimizer, tokenizer, scheduler

# from https://discuss.huggingface.co/t/keeping-iterabledataset-node-wise-split-fixed-during-ddp/58713/3
def get_stream_dataset(config):
    ds = load_dataset("c4", "en", split = 'train', streaming = True)
    ds = ds.shuffle(seed = 42, buffer_size = config.buffer_size)
    ds = split_dataset_by_node(ds, world_size = int(os.environ["WORLD_SIZE"]), rank = int(os.environ["RANK"]))
    ds = ds.map(tokenize, batched = True, remove_columns = ['text', 'timestamp', 'url', 'attention_mask'])
    ds = ds.with_format(type = 'torch')
    return ds

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers = 1
    )

def main(config):
    ddp_setup(config)
    run = None
    if config.use_wandb and int(os.environ["RANK"]) == 0:
        import wandb
        wandb.login()
        run = wandb.init(project=config.wandb_project, name = config.wandb_name, config = config)
    model, optimizer, tokenizer, scheduler = load_train_objs(config)
    dset = get_stream_dataset(config)
    train_data = prepare_dataloader(dset, config.batch_size)
    val_data = None
    trainer = Trainer(model, train_data, val_data, optimizer, scheduler, tokenizer, config, dset, run, grad_accumulate=config.grad_accumulate)
    trainer.train()
    if config.use_wandb and int(os.environ["RANK"]) == 0:
        wandb.finish()
    destroy_process_group()

if __name__ == "__main__":
    from config import config
    print('HELLO')
    main(config)