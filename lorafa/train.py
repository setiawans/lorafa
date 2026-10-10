import torch
import torch.nn.functional as F

def run_epoch(model, loader, dev, optimizer=None, scaler=None, amp=False, freeze_bn_stats=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn_stats:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            count += y.size(0)
    return total_loss / count, 100.0 * correct / count

@torch.no_grad()
def evaluate(model, loader, dev):
    return run_epoch(model, loader, dev)