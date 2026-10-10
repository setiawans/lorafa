import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.checkpoints import load_checkpoint, load_lorafa, save_checkpoint  # noqa: E402
from lorafa.data import get_dataset, get_loader  # noqa: E402
from lorafa.train import evaluate  # noqa: E402
from lorafa.utils import device  # noqa: E402

SRC = Path("checkpoints/finetune/resnet20_4_r16")
OUT = Path("checkpoints/finetune/resnet20_4_r16_splice")
SPLICES = [("fc1_b0", SRC / "epoch000.pt", SRC / "epoch001.pt"), ("fc0_b1", SRC / "epoch001.pt", SRC / "epoch000.pt")]

def splice(name: str, b_source: Path, fc_source: Path, dev) -> None:
    model, meta, norm = load_lorafa(b_source, dev)
    fc_state = load_checkpoint(fc_source)[0]
    model.fc.load_state_dict({k[3:]: v for k, v in fc_state.items() if k.startswith("fc.")})
    test_loader = get_loader(get_dataset("cifar10", "test", "data", norm), 256, False, 0)
    te_loss, te_acc = evaluate(model, test_loader, dev)
    save_checkpoint(OUT / f"{name}.pt", model, {**meta, "B_source": str(b_source), "fc_source": str(fc_source),
                                                 "test_loss": round(te_loss, 4), "test_acc": round(te_acc, 2)})

def main():
    dev = device()
    for name, b_source, fc_source in SPLICES:
        splice(name, b_source, fc_source, dev)

if __name__ == "__main__":
    main()