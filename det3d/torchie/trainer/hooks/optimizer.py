from torch.nn.utils import clip_grad
import torch.cuda.amp as amp

from .hook import Hook


class OptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip
        self.scaler = amp.GradScaler()

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip
        )

    def after_train_iter(self, trainer):
        trainer.optimizer.zero_grad()
        # print(trainer.outputs["loss"])
        self.scaler.scale(trainer.outputs["loss"]).backward()
        if self.grad_clip is not None:
            self.clip_grads(trainer.model.parameters())
        self.scaler.step(trainer.optimizer)
        self.scaler.update()
