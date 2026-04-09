import torch

from .plot import visualize_predictions
from ..utils import Cholec80, FEATURE_SEQ, PADDING_MASK, LABEL_SEQ, LOGITS, calculate_metrics


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

        input = batch[FEATURE_SEQ]
        N, S, C = input.shape
        results["batch_size"] = N
        input = input.to(device_gpu)

        in_valid_mask = torch.logical_not(batch[PADDING_MASK])
        if train is False:
            results["in_valid_mask_cpu"] = in_valid_mask.detach()
        nelems = in_valid_mask.sum().item()
        in_valid_mask = in_valid_mask.to(device_gpu)

        if get_target is True:
            target = batch[LABEL_SEQ]
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

        logits = out[LOGITS][-1]
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
    def __init__(self, deep_supervision, phase_recognition_loss, smooth_logits_loss,
                 phase_recognition_factor, smooth_logits_factor, device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        self.dataset = dataset
        self.deep_supervision = deep_supervision

        self.loss_functions = {
            "phase_recognition": phase_recognition_loss,
            "smooth_logits": smooth_logits_loss,
        }

        self.loss_factors = {
            "phase_recognition": phase_recognition_factor,
            "smooth_logits": smooth_logits_factor
        }

        self.loss_keys = ["loss_phase_recognition"]
        if smooth_logits_factor > 0:
            self.loss_keys.append("loss_smooth")

        self.device_cpu = device_cpu
        self.device_gpu = device_gpu

        self.predictions = None
        self.nplot = nplot

    def get_loss_keys(self):
        return self.loss_keys

    def process_batch(self, model, batch, train=True):
        """Compute losses and model predictions given a batch from the dataloader."""
        input, in_valid_mask, target, nelems, N, results = model.parse_batch(batch, self.device_gpu, train)

        if model.channels_last is False:
            input = input.permute(0, 2, 1)  # N x S x C --> N x C x S
        out = model(input, in_valid_mask)

        self.compute_standard_loss(
            out[LOGITS], in_valid_mask, target, nelems, N, results, train, model.channels_last
        )

        return results

    def compute_standard_loss(self, all_logits, in_valid_mask, target, nelems, N, results, train=True, channels_last=True):
        total_loss = 0
        loss_factor_sum = 0

        nlogits = len(all_logits)
        for i in range(0 if self.deep_supervision else (nlogits - 1), nlogits):
        # assumes that 'refinement' of predicted logits increases with index i
        # --> logits at final index correspond to overall best predictions
            logits = all_logits[i]

            if channels_last is True:
                logits_chF = logits.permute(0, 2, 1)  # N x S x C --> N x C x S; channels first format
                logits_chL = logits  # N x S x C; channels last format
            else:
                logits_chF = logits  # N x C x S
                logits_chL = logits.permute(0, 2, 1)  # N x C x S --> N x S x C; channels last format
            del logits

            loss_ = self.loss_functions["phase_recognition"](logits_chF, target) * self.loss_factors["phase_recognition"]
            total_loss = total_loss + loss_
            loss_factor_sum += self.loss_factors["phase_recognition"]

            if "loss_phase_recognition" not in results:
                results["loss_phase_recognition"] = [loss_.item(), nelems]
            else:
                results["loss_phase_recognition"][0] = results["loss_phase_recognition"][0] + loss_.item()
                results["loss_phase_recognition"][1] = results["loss_phase_recognition"][1] + nelems

            if self.loss_factors["smooth_logits"] > 0:
                loss_ = self.loss_functions["smooth_logits"](logits_chL, in_valid_mask) * self.loss_factors["smooth_logits"]
                total_loss = total_loss + loss_
                loss_factor_sum += self.loss_factors["smooth_logits"]

                if "loss_smooth" not in results:
                    results["loss_smooth"] = [loss_.item(), nelems - N]
                else:
                    results["loss_smooth"][0] = results["loss_smooth"][0] + loss_.item()
                    results["loss_smooth"][1] = results["loss_smooth"][1] + (nelems - N)

        total_loss = total_loss / loss_factor_sum  # normalize loss factors

        if train is True:
            results["total_loss"] = total_loss

        self.get_prediction(logits_chF, in_valid_mask, target, results, nelems, train)

    @staticmethod
    def get_prediction(logits_chF, in_valid_mask, target, results, nelems, train=True):
        with torch.no_grad():
            _, predicted = torch.max(torch.nn.Softmax(dim=1)(logits_chF), dim=1)
            if train:  # estimate prediction accuracy
                correct = ((predicted == target) * in_valid_mask).sum().item()
                results["accuracy"] = (correct / nelems, nelems)
            else:
                results["predicted_gpu"] = predicted.detach()

    def reset(self, train=False):
        self.predictions = {}

    def update_predictions(self, results, train=True):
        """Keep track of predictions to create visualizations."""
        if train is True:
            pass
        else:
            predicted = results["predicted_gpu"].to(self.device_cpu).numpy()
            target = results["target_cpu"].numpy()
            valid_mask = results["in_valid_mask_cpu"].numpy()

            for i in range(results["batch_size"]):
                P = predicted[i, valid_mask[i, :]]
                Y = target[i, valid_mask[i, :]]

                metrics = calculate_metrics(Y, P, self.dataset.phase_labels)
                key_ = ("{:.4f}".format(metrics["accuracy"]), "{:.4f}".format(metrics["macro_jaccard"]))
                self.predictions[key_] = {
                    'predicted': P,
                    'target': Y,
                }

    def visualize_outputs(self, epoch, logger, logger_prefix=None, train=True):
        if train is True:
            pass
        else:
            assert (logger_prefix is not None)
            achieved_metrics = sorted(sorted(list(self.predictions.keys()), key=lambda t: t[0]), key=lambda t: t[1])
            if len(achieved_metrics) > 0:
                # show results with lowest performance
                to_plot = [self.predictions[key_] for key_ in achieved_metrics[:self.nplot]]
                logger.add_figure("{}/predictions".format(logger_prefix), visualize_predictions(to_plot, self.dataset.num_phases), epoch)
