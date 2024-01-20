import torch
import numpy as np
from yolov5.utils.general import non_max_suppression, box_iou
from lib.segmentation import SuperPixler
import shap
import matplotlib as mpl
from lib.image import img_float_to_uint
from lib.segmentation import fill_segmentation
import matplotlib.pyplot as plt


class CastNumpy(torch.nn.Module):
    def __init__(self, to: str, device: torch.device, half: bool = False):
        super(CastNumpy, self).__init__()
        self.half = half
        self.to = to
        self.device = device
        assert to in set(("numpy", "torch"))

    def forward(self, image):
        """
        In the forward function we accept the inputs and cast them to a pytorch tensor
        """

        image = np.ascontiguousarray(image)

        if self.to == "numpy":
            return image
        elif self.to == "torch":
            image = torch.from_numpy(image).to(self.device)

        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        if self.half is True:
            return image.half()

        return image


class OD2Score(torch.nn.Module):
    """
    given target bbox, nms conf threshold, iou_threshold and predictions
    from model performes nms and reuturns score* base on prediction with
    best fitting bbox. If there are no predictions score is 0.

    bbox*, conf* - best fitting bbox and its score
    result = iou(target, bbox*) * conf*
    """

    def __init__(
        self,
        target_bbox: np.ndarray,
        device: torch.device,
        conf_thresh: float = 0.01,
        iou_thresh: float = 0.5,
    ):
        super(OD2Score, self).__init__()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_bbox = torch.tensor(
            target_bbox, device=device
        )  # (1, 4) of (x1, y1, x2, y2)
        assert self.target_bbox.dim() == 2
        assert self.target_bbox.shape == (1, 4)

    def forward(self, prediction: torch.tensor):
        """
        In the forward function we accept the predictions and return the score for a selected target of the box
        prediction: (batch_size, n_detections, 6), where last dim has structure of (x1, y1, x2, y2, pred_score, pred_cls)

        (batch_size, n_detections, 6) -> (batch_size)
        """

        assert prediction.dim() == 3
        assert prediction.shape[2] == 6
        filtered = non_max_suppression(
            prediction, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
        )  #  List of [tensor(n_detections, 6)] * batch_size

        scores = torch.zeros(len(filtered), device=prediction.device)

        for idx, batch in enumerate(filtered):
            # if there are no detections set score to 0
            if batch.shape[0] == 0:
                continue
            scores[idx] = (box_iou(self.target_bbox, batch[:, :4]) * batch[:, 4]).max()

        return scores


class Permute(torch.nn.Module):
    def __init__(self, *permutation):
        super(Permute, self).__init__()
        self.permutation = tuple(permutation)

    def forward(self, x):
        return x.permute(self.permutation)


class BatchedProcessing(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, batch_size: int = 64):
        super(BatchedProcessing, self).__init__()
        self.batch_size = batch_size
        self.model = model

    def _batch_division(self, n_elems: int, batch_size: int):
        last = 0
        curr = min(batch_size, n_elems)

        while curr <= n_elems:
            yield last, curr
            last = curr
            curr = curr + batch_size

        if last < n_elems:
            yield last, n_elems

    def forward(self, x: torch.Tensor):
        # assumes that first dim is batch size
        results = []
        total_elems = x.shape[0]

        for start, end in self._batch_division(total_elems, self.batch_size):
            r = self.model(x[start:end])
            results.append(r)

        return np.concatenate(results, axis=0)


def calculate_shap(
    model,
    super_pixler: SuperPixler,
    bbox_target: torch.Tensor,
    nsamples: int = 80,
    batch_size: int = 256,
    half: bool = False,
) -> np.ndarray:
    background_super_pixel = np.ones((1, super_pixler.num_segments))
    image_super_pixel = np.zeros_like(background_super_pixel)
    device = model.model.device

    patch_scorer = torch.nn.Sequential(
        super_pixler,  # (n_outputs, H, W, C) of float numpy
        CastNumpy(
            to="torch", device=device, half=half
        ),  # (n_outputs, H, W, C) of float torch
        Permute(0, 3, 1, 2),  # (n_outputs, C, H, W) of float torch
        model,  # (n_outputs, num_detections, 6) of float torch
        OD2Score(
            bbox_target, device, model.iou
        ),  # (n_outputs, n_detections, 6) -> (n_outputs) of float torch,
        CastNumpy(to="numpy", device=device),  # (n_outputs, n_segments) of float numpy
    )

    batched_ps = BatchedProcessing(
        model=patch_scorer, batch_size=batch_size
    )  # (batch_size,) of numpy array,

    kernel_explainer = shap.KernelExplainer(batched_ps, background_super_pixel)
    shap_values = kernel_explainer.shap_values(image_super_pixel, nsamples=nsamples)
    return shap_values


def plot_shap_explenations(
    img: np.ndarray,
    shap_values: np.ndarray,
    segments: np.ndarray,
    blend_alpha: float = 0.5,
):
    assert img.ndim == 3
    assert img.shape[2] == 3

    assert segments.ndim == 2
    assert segments.shape[0] == img.shape[0]
    assert segments.shape[1] == img.shape[1]

    assert shap_values.ndim == 1
    assert shap_values.shape[0] == segments.max() + 1

    absmax = np.abs(shap_values).max()

    cm = mpl.colormaps["seismic"]
    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-absmax, vmax=absmax)
    shap_val_mask = fill_segmentation(shap_values, segments)

    fig, ax = plt.subplots(ncols=1)
    ax.imshow(img_float_to_uint(img), alpha=1 - blend_alpha)
    im = ax.imshow(shap_val_mask, cmap=cm, norm=norm, alpha=blend_alpha)
    ax.axis("off")
    cb = fig.colorbar(
        im, ax=ax, norm=norm, label="SHAP value", orientation="horizontal", aspect=60
    )
    cb.outline.set_visible(False)

    plt.show()
