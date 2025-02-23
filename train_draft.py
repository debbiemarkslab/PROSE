from torch.utils.data import Dataset, DataLoader
from poet.alphabets import Alphabet
import numpy as np
import torch
import argparse
import tqdm 
from poet.models.poet import PoET

class ATCG(Alphabet):
    ''' tokenize DNA sequence like this? 
    '''
    def __init__(
        self,
        mask=False,
        include_gap=False,
        include_startstop=False,
        distinct_startstop=False,
    ):
        chars = b"ATCG"
        gap_token = start_token = stop_token = -1
        if include_gap:
            chars = chars + b"-"
            gap_token = len(chars) - 1
        if include_startstop:
            chars = chars + b"*"
            start_token = stop_token = len(chars) - 1
        if distinct_startstop:
            chars = chars + b"$"
            stop_token = len(chars) - 1
        mask_token = len(chars) 
        encoding = np.arange(len(chars))
        missing = mask_token

        super(ATCG, self).__init__(
            chars, encoding=encoding, mask=mask, missing=missing
        )

        self.gap_token = gap_token
        self.start_token = start_token
        self.stop_token = stop_token
        self.mask_token = mask_token


class PromoterDataset(Dataset):
    ''' Still need to figure out how each family is organized
    '''
    def __init__(self, sequences, alphabet):
        self.alphabet = alphabet
        self.sequences = [self._process_seq(s) for s in sequences]

    def _process_seq(self, seq):
        return [self.alphabet.start_token] + self.alphabet.encode(seq)+ [self.alphabet.stop_token]

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = self._process_seq(seq)
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor([len(tokens)], dtype=torch.int)
        )
    def __len__(self):
        return len(self.sequences)

def collate_fn(batch):
    xs, lengths = zip(*batch)
    padded_xs = torch.nn.utils.rnn.pad_sequence(
        xs, batch_first=True, padding_value=alphabet.mask_token
    )
    return padded_xs, torch.tensor(lengths)

class SqrtDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    ''' they claim they use this in the paper,
        couldn't find any info on this so this is fully AI-generated
    '''
    def __init__(self, 
                 optimizer: torch.optim.optimizer.Optimizer,
                 last_epoch: int = -1,
                 verbose: bool = False,
                 scaling_factor: float = 1.0,
                 offset: int = 0):
        
        self.scaling_factor = scaling_factor
        self.offset = offset
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> list[float]:
        """Compute scaled learning rate using sqrt decay"""
        return [base_lr * self.scaling_factor / torch.sqrt(self.last_epoch + 1 + self.offset)
                for base_lr in self.base_lrs]

    def theoretical_bound(self, 
                         total_steps: int,
                         gradient_bound: float) -> float:
        """
        Theoretical convergence bound for convex functions with L-Lipschitz gradients:
        
        f(x̄_T) - f(x^*) ≤ (2D²L + D√(2σ²T)) / √T
        
        Where:
        - D: Diameter of feasible set
        - L: Lipschitz constant
        - σ²: Gradient variance bound
        - T: Total number of steps
        
        Returns expected optimization error bound
        """
        return (2 * self.scaling_factor**2 * gradient_bound**2 + 
                self.scaling_factor * torch.sqrt(2 * gradient_bound**2 * total_steps)) / torch.sqrt(total_steps)


def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        xs, _ = batch
        xs = xs.to(device)
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type="cuda"):
            logits = model(xs, segment_sizes=None)
            targets = xs[:, 1:]
            logits = logits[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=alphabet.mask_token
            )
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            xs, _ = batch
            xs = xs.to(device)
            
            logits = model(xs, segment_sizes=None)
            targets = xs[:, 1:]
            logits = logits[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=alphabet.mask_token
            )
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main(args):
    # TODO: nothing is parallelized yet, should probably try the Fabric thing ruben used
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    
    # Model initialization
    model = PoET(
        n_vocab=len(alphabet),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.n_heads,
        dropout=args.dropout
    ).to(device)
    
    # this is mentioned in the paper
    optimizer = torch.optim.Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = SqrtDecayScheduler(
        optimizer, )
    scaler = torch.amp.GradScaler()

    # TODO: Data loading
    train_seqs = [...]  # Load training sequences
    val_seqs = [...]    # Load validation sequences
    
    train_dataset = PromoterDataset(train_seqs, alphabet)
    val_dataset = PromoterDataset(val_seqs, alphabet)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Training loop
    for epoch in range(args.epochs):

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        val_loss = validate(model, val_loader, device)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f"checkpoint_epoch_{epoch}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--distributed", action="store_true")
    
    args = parser.parse_args()
    
    alphabet = ATCG(
        include_gap=True,
        include_startstop=True,
        distinct_startstop=True
    )

    
