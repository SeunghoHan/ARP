import numpy as np

def pixel_accuracy(true_mask, pred_mask):
    """
    Compute the pixel accuracy between the true mask and the predicted mask.

    Parameters:
    - true_mask: numpy array of ground truth labels
    - pred_mask: numpy array of predicted labels

    Returns:
    - accuracy: pixel accuracy
    """
    assert true_mask.shape == pred_mask.shape, "Shape of true mask and predicted mask must be the same"
    
    correct = np.sum(true_mask == pred_mask)
    total = true_mask.size
    
    accuracy = correct / total
    return accuracy

# Example usage
true_mask = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]])
pred_mask = np.array([[1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0]])

accuracy = pixel_accuracy(true_mask, pred_mask)
return (f"Pixel Accuracy: {accuracy:.4f}")