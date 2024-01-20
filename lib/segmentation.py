import torch
import numpy as np
from skimage.segmentation import slic, mark_boundaries


class SuperPixler(torch.nn.Module):
  def __init__(self, image, n_segments: int = 100, sigma: int = 0, compactness: float = 10.0):
    super(SuperPixler, self).__init__()

    self.image = image  # must be passed in float dtype scale [0, 1]
    assert image.shape[2] == 3
    self.mean_color = self.image.mean()
    self.mean_img = np.zeros_like(image) + self.mean_color
    self.segments = slic(image, n_segments=n_segments, sigma=sigma, compactness=compactness) - 1
    self.num_segments = self.segments.max().item() + 1


  def mark_boundaries(self):
      return mark_boundaries(self.image, self.segments)

  def get_segments(self):
      return self.segments.copy()

  def forward(self, masks: np.array):
    """
    In the forward step we accept the super pixel masks and transform them to a batch of images
    True - apply mask, False - do not apply mask
    """

    masks = masks.astype(bool)
    assert len(masks.shape) == 2
    assert masks.dtype == bool
    assert masks.shape[1] == self.num_segments

    # outputs are not stacking images
    n_outputs = masks.shape[0]
    outputs = np.stack(tuple(self.image.copy() for _ in range(n_outputs)), axis=0)

    for output_idx in range(n_outputs):
      for segment_idx in range(self.num_segments):
        if masks[output_idx][segment_idx].item() == True:
            # segments take value from [1, num_segments]
            m = (self.segments == segment_idx)
            outputs[output_idx][m] = self.mean_color

    # output is of shape (n_outputs, w, h, 3)
    return outputs


def fill_segmentation(values, segmentation: np.ndarray):
    out = np.zeros_like(segmentation, dtype=float)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out
