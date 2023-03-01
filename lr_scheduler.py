import torch
import matplotlib.pyplot as plt


class TrainingScheduler(object):
    """Gère le warmup et le scheduler de learning rate"""

    def __init__(self, optimizer, lr0: float, lr_scheduler, warmup_iteration: int = 10):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.lr0 = lr0
        self.lr_scheduler = lr_scheduler
        if self.warmup_iteration > 0:
            self.step(0.5)  # On a un lr de ((lr0 * 0.5) / warmup_iteration) à l'epoch 0

    def warmup(self, cur_iteration):
        warmup_lr = self.lr0 * float(cur_iteration) / float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration < self.warmup_iteration:
            self.warmup(cur_iteration)
        else:
            self.lr_scheduler.step()

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)


if __name__ == "__main__":
    v = torch.zeros(10)
    lr = 0.01
    total_iter = 100
    warmup_iter = 10

    optim = torch.optim.SGD([v], lr=lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_iter - warmup_iter
    )
    scheduler = TrainingScheduler(
        optimizer=optim,
        warmup_iteration=warmup_iter,
        lr0=lr,
        lr_scheduler=scheduler_cosine,
    )

    x_iter = [0]
    y_lr = [0.0]

    for iter in range(1, total_iter + 1):
        print("iter: ", iter, " ,lr: ", optim.param_groups[0]["lr"])

        optim.zero_grad()
        optim.step()

        scheduler.step(iter)

        x_iter.append(iter)
        y_lr.append(optim.param_groups[0]["lr"])

    plt.plot(x_iter, y_lr, "b")
    plt.legend(["learning rate"])
    plt.xlabel("iteration")
    plt.ylabel("learning rate")
    plt.show()
