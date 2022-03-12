import keras
from keras import losses
import keras.backend as K
import numpy


def DiceLoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    # inputs = K.flatten(inputs)
    # targets = K.flatten(targets)

    y_multiplication = targets * inputs
    intersection = K.sum(y_multiplication, axis=-1)
    denominator = K.sum(K.square(targets), axis=-1) + K.sum(K.square(inputs), axis=-1)
    dice = (2. * intersection + smooth) / (denominator + smooth)
    return 1 - dice


def DiceBCELoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = keras.losses.BinaryCrossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss

    return Dice_BCE


# ne marche pas du tout
# def dice_coef_loss(y_true, y_pred):
#     def dice_coef(y_true, y_pred, smooth=1e-6):
#         """
#         Dice = (2*|X & Y|)/ (|X|+ |Y|)
#              =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#         ref: https://arxiv.org/pdf/1606.04797v1.pdf
#         """
#         intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#         return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
#     return 1-dice_coef(y_true, y_pred)



def dice_coef_loss(y_true, y_pred):
    smooth = 1.  # Used to prevent denominator 0

    def dice_coef(y_true, y_pred):
        print(y_true)
        y_true_f = K.flatten(y_true)  # y_true stretch to one dimension
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

    return 1. - dice_coef(y_true, y_pred)


def IoULoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU


ALPHA = 0.8
GAMMA = 2


def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return focal_loss


ALPHA = 0.5
BETA = 0.5


def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
    # flatten label and prediction tensors
    # inputs = K.flatten(inputs)
    # targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - Tversky


def tversky_index(y_true, y_pred):
    # generalization of dice coefficient algorithm
    #   alpha corresponds to emphasis on False Positives
    #   beta corresponds to emphasis on False Negatives (our focus)
    #   if alpha = beta = 0.5, then same as dice
    #   if alpha = beta = 1.0, then same as IoU/Jaccard
    alpha = 0.5
    beta = 0.5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection) / (intersection + alpha * (K.sum(y_pred_f*(1. - y_true_f))) + beta *  (K.sum((1-y_pred_f)*y_true_f)))


def tversky_index_loss(y_true, y_pred):
    return -tversky_index(y_true, y_pred)


ALPHA = 0.5
BETA = 0.5
GAMMA = 1


def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    FocalTversky = K.pow((1 - Tversky), gamma)

    return FocalTversky


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1,2,3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def iou_coef_loss(y_true, y_pred):
    return 1-iou_coef(y_true, y_pred)


def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
