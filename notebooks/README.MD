# Google adversarial patch experiments

We replace a small part of the image with a patch that is adversarial to the YOLOv5 model. The expected result would be that model cannot detect the object in the image or it's confidence in detection decreases significantly.

### First try
[notebook](adversarial_patch_1.ipynb)

This is the first version of patch training. Key points:
1. Patch is a square of (N, M) pixels
2. Training is done with mini batches
3. At each iteration the augumented patch is applied to the whole batch at random (x, y) coordinates
4. Augumentation is just scaling for now
5. The optimized objective is the sum of all confidences for all outputed bounding boxes in the batch


The training loss is very noisy which is intuitively expected (?) since we put patch in random places and then expect the model to change it's confidences for all found bounding boxes (even for grid cells which do not even "see" the patch). As I understand YOLO at first the model is classifying each grid cell and putting the patch in grid cell A shouldn't change the classification output in grid cell B.

The loss doesn's seem to drop after 10 epochs. Maybe just keeping the training will eventually cause some breakthrough - my intuition is that if we tell model to drop confidence for a bounding box which doesn't include nor is near the license plate, then the confidence was small anyway (because the model is good at recognizing things) so we don't introduce that much noise. But if we by chance put the patch in the right place of the photo (near license plate) and tell model to drop it's confidence for the bounding box which includes the license plate, then telling the model to drop the confidence there would be a strong, good learning signal.


## Ideas to explore
- Different patch augumentations/shapes
- Put patches in the right place of the photo (near license plate) during patch training
- Put a lot of patches in the image at once during training
- Somehow filter for which bouding boxes we want to drop confidence. For each batch we can the model without the patch, get the indexes of bounding boxes which include the license plate/have high confindence above some threshold.
- Use another objective? For example to minimize the size of produced bounding boxes, or minimize their coordinates. Anything to make the model disoriented.
- Find out how many grid cells are used in YOLOv5. Then we could:
    - Place the small patch in the corner of each grid cell. This could cause model to misclassify the grid cells. Training the patch for misclassification an image is done by Google Adversarial Patch paper so it should work.
