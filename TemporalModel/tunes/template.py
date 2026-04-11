import torch


class ModelTemplate:
    def __init__(self, channels_last=True):
        self.channels_last = channels_last  # processes tensors of shape N x S x C (otherwise: N x C x S)

    def setup(self):
        pass

    def get_temporal_scales(self):
        return []

    @staticmethod
    def parse_batch(batch, device_gpu, train=True, get_target=True):
        results = {}

        input = batch["feature_seq"]
        N, S, C = input.shape
        results["batch_size"] = N
        input = input.to(device_gpu)

        in_valid_mask = torch.logical_not(batch["padding_mask"])
        if train is False:
            results["in_valid_mask_cpu"] = in_valid_mask.detach()
        nelems = in_valid_mask.sum().item()
        in_valid_mask = in_valid_mask.to(device_gpu)

        if get_target is True:
            target = batch["label_seq"]
            if train is False:
                results["target_cpu"] = target.detach()
            target = target.to(device_gpu)
        else:
            target = None

        return input, in_valid_mask, target, nelems, N, results

    def infer_batch(self, batch, device_gpu, return_logits=False):
        """Compute model predictions given a batch from the dataloader."""
        input, in_valid_mask, _, _, _, results = self.parse_batch(batch, device_gpu, train=False, get_target=False)

        if self.channels_last is False:
            input = input.permute(0, 2, 1)  # N x S x C --> N x C x S
        out = self.forward(input, in_valid_mask)

        logits = out["logits"][-1]
        if return_logits:
            if self.channels_last:
                results["logits_gpu"] = logits.detach()
            else:
                results["logits_gpu"] = logits.permute(0, 2, 1).detach()
        else:
            cdim = -1 if self.channels_last else 1
            _, predicted = torch.max(torch.nn.Softmax(dim=cdim)(logits), dim=cdim)
            results["predicted_gpu"] = predicted.detach()

        return results

    def dummy_forward(self, input, valid_mask):  # shapes are input: N x S x C, valid_mask: N x S; both are GPU tensors
        if self.channels_last is False:
            input = input.permute(0, 2, 1)  # N x S x C --> N x C x S
        self.forward(input, valid_mask)


class TrainerTemplate:
    pass
