import os
import torch
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from itertools import permutations
from sklearn.neighbors import NearestNeighbors

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from reformer.reformer_pytorch import ViRWithArcMargin

import os
import random
import argparse
import time
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import umap
from scipy.spatial import ConvexHull
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Define the transformations for the dataset
test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def save_features(features, save_path):
    """
    Save extracted features to a file.
    
    Args:
        features (dict): Dictionary mapping file paths to feature vectors
        save_path (str): Path where features will be saved
    """
    paths = list(features.keys())
    feature_vectors = np.array([features[path] for path in paths])
    np.savez(save_path, features=feature_vectors, paths=paths)
    print(f"Features saved to {save_path}")

def load_features(load_path):
    """
    Load features from a saved file.
    
    Args:
        load_path (str): Path to the saved features file
    
    Returns:
        dict: Dictionary mapping file paths to feature vectors
    """
    data = np.load(load_path, allow_pickle=True)
    features = {}
    for path, feature_vector in zip(data['paths'], data['features']):
        features[str(path)] = feature_vector
    return features

def get_unique_paths_from_pairs(pair_list):
    """
    Extract unique image paths from the pair list file.
    
    Args:
        pair_list (str): Path to the file containing image pairs
        
    Returns:
        set: Set of unique image paths
    """
    unique_paths = set()
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    for pair in pairs:
        splits = pair.strip().split()
        path1, path2 = splits[0], splits[1]
        unique_paths.add(path1)
        unique_paths.add(path2)
    return unique_paths

def get_features_from_images(model, pair_list, save_path=None):
    """
    Extract features from images listed in the pair file and return them in a dictionary.
    
    Args:
        model: The neural network model
        pair_list (str): Path to the file containing image pairs
        save_path (str, optional): If provided, save features to this path
    
    Returns:
        dict: Dictionary mapping file paths to feature vectors
    """
    image_paths = get_unique_paths_from_pairs(pair_list)
    print(f"Processing {len(image_paths)} unique images from pair list")
    
    features = {}
    model.eval()
    with torch.no_grad():
        for image_path in image_paths:
            if image_path in features:
                continue
            image = Image.open(image_path).convert('RGB')
            image = test_transform(image).unsqueeze(0).cuda()
            output = model.extract_features(image)
            output = output.data.cpu().numpy()
            features[image_path] = output[0]

    if save_path:
        save_features(features, save_path)

    return features

def load_model(model, model_path):
    """
    Load pretrained model weights.
    
    Args:
        model: The neural network model
        model_path (str): Path to the pretrained weights file
    """
    print(f"Loading model from {model_path}")
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def cosine_metric(x1, x2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        x1 (numpy.ndarray): First vector
        x2 (numpy.ndarray): Second vector
    
    Returns:
        float: Cosine similarity score
    """
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_accuracy(y_score, y_true):
    """
    Calculate accuracy and find the best threshold.
    
    Args:
        y_score (list): List of similarity scores
        y_true (list): List of true labels
    
    Returns:
        tuple: (best_accuracy, best_threshold)
    """
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return (best_acc, best_th)

def test_performance(features, pair_list):
    """
    Test performance using cosine similarity between pairs of features.
    
    Args:
        features (dict): Dictionary mapping file paths to their feature vectors.
        pair_list (str): Path to the file containing pairs to test.
    
    Returns:
        tuple: (accuracy, threshold)
    """
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    
    for pair in pairs:
        splits = pair.strip().split()
        path1, path2, label = splits[0], splits[1], int(splits[2])
        fe_1 = features[path1]
        fe_2 = features[path2]
        sim = cosine_metric(fe_1, fe_2)
        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th

def nearest_neighbor_metric(features, n_neighbors=5):
    """
    Calculate nearest neighbor distances for each feature vector.
    
    Args:
        features (numpy.ndarray): Array of feature vectors
        n_neighbors (int): Number of neighbors to consider
    
    Returns:
        numpy.ndarray: Array of distances to the nearest neighbors
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(features)
    distances, _ = nn.kneighbors(features)
    return distances

def test_performance_with_nn(features, pair_list, n_neighbors=5):
    """
    Test performance using nearest neighbor distances between pairs of features.
    
    Args:
        features (dict): Dictionary mapping file paths to their feature vectors.
        pair_list (str): Path to the file containing pairs to test.
        n_neighbors (int): Number of neighbors to consider
    
    Returns:
        tuple: (accuracy, threshold)
    """
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    paths = list(features.keys())
    feature_vectors = np.array([features[path] for path in paths])

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(feature_vectors)
    distances, _ = nn.kneighbors(feature_vectors)

    sims = []
    labels = []
    
    for pair in pairs:
        splits = pair.strip().split()
        path1, path2, label = splits[0], splits[1], int(splits[2])
        idx1 = paths.index(path1)
        idx2 = paths.index(path2)
        sim = np.mean(distances[idx1]) + np.mean(distances[idx2])
        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th

def is_black_image(image_path, threshold=0.95):
    """
    Check if an image is predominantly black.

    Args:
        image_path (str): Path to the image file.
        threshold (float): Threshold for considering an image as black. 
                           If the proportion of black pixels is greater than this, the image is considered black.

    Returns:
        bool: True if the image is predominantly black, False otherwise.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_array = np.array(image)
    
    # Calculate the proportion of black pixels
    black_pixels = np.sum(image_array < 10)  # Pixels with value less than 10 are considered black
    total_pixels = image_array.size
    black_ratio = black_pixels / total_pixels
    
    return black_ratio > threshold

def remove_black_images(dataset_root, threshold=0.95):
    """
    Remove black images from the dataset.

    Args:
        dataset_root (str): Root directory of the dataset.
        threshold (float): Threshold for considering an image as black.
    """
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                if is_black_image(image_path, threshold):
                    print(f"Removing black image: {image_path}")
                    os.remove(image_path)

def visualize_with_umap(features, num_people=5, random_seed=42, save_path=None):
    """
    Enhanced UMAP visualization with image thumbnails and class regions.
    
    Args:
        features (dict): A dictionary mapping image paths to feature vectors
        num_people (int): Number of random people to select for visualization
        random_seed (int): Random seed for reproducibility
        save_path (str or None): If provided, saves the plot to this path. If None, the plot is shown
    """
    np.random.seed(random_seed)

    # Extract and process names
    file_paths = list(features.keys())
    person_names = []
    for path in file_paths:
        parts = path.split(os.sep)
        for i, part in enumerate(parts):
            if part == "lfw_funneled" and i + 1 < len(parts):
                person_names.append(parts[i + 1])
                break
        else:
            person_names.append(parts[-2])

    # Get unique names and their counts
    name_counts = {}
    for name in person_names:
        name_counts[name] = name_counts.get(name, 0) + 1

    # Filter to include only people with multiple images
    min_images = 3
    qualified_names = [name for name, count in name_counts.items() if count >= min_images]

    if len(qualified_names) < num_people:
        print(f"Warning: Only {len(qualified_names)} people have {min_images}+ images. Using all of them.")
        selected_names = qualified_names
    else:
        selected_names = np.random.choice(qualified_names, size=num_people, replace=False)

    # Filter features
    filtered_features = []
    filtered_names = []
    filtered_paths = []
    for path, feature in features.items():
        person_name = None
        parts = path.split(os.sep)
        for i, part in enumerate(parts):
            if part == "lfw_funneled" and i + 1 < len(parts):
                person_name = parts[i + 1]
                break
        else:
            person_name = parts[-2]

        if person_name in selected_names:
            filtered_features.append(feature)
            filtered_names.append(person_name)
            filtered_paths.append(path)

    # Convert to numpy array and normalize
    feature_vectors = np.array(filtered_features)
    feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)

    # Apply UMAP
    print("Applying UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_features = reducer.fit_transform(feature_vectors)

    # Plotting
    plt.figure(figsize=(20, 15))

    unique_filtered_names = list(set(filtered_names))
    num_colors = len(unique_filtered_names)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
    name_to_color = dict(zip(unique_filtered_names, colors))

    ax = plt.gca()

    # Plot individual points with thumbnails
    for (x, y), name, path in zip(reduced_features, filtered_names, filtered_paths):
        img = plt.imread(path)
        thumbnail = OffsetImage(img, zoom=0.1)
    
        # Add colored background
        bg_color = name_to_color[name]
        ab = AnnotationBbox(thumbnail, (x, y), frameon=True, 
                            fontsize=10, 
                            bboxprops=dict(facecolor=bg_color, alpha=0.5))
    
        ax.add_artist(ab)
        plt.scatter(x, y, c=bg_color, alpha=0.8)

    plt.title(f"")
    plt.xlabel("UMAP component 1")
    plt.ylabel("UMAP component 2")

    # Sort by count and create legend
    selected_names_sorted = sorted(selected_names, key=lambda x: name_counts[x], reverse=True)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=name_to_color[name], 
                                label=f"{name} ({name_counts[name]} images)", 
                                markersize=10)
                      for name in selected_names_sorted]

    #plt.legend(handles=legend_elements, 
    #          title="People (Image Count)",
    #          bbox_to_anchor=(1.05, 1),
    #          loc='upper left',
    #          borderaxespad=0.)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

def generate_lfw_pair_list(dataset_root, pair_file, num_pairs=50):
    """
    Generate a pair list for LFW testing, restricting paths to the 'testing' directory.

    Args:
        dataset_root (str): Root directory of the LFW dataset
        pair_file (str): File to save the generated pair list
        num_pairs (int): Number of pairs to generate
    """
    label_to_images = {}

    # Collect image file paths by label within the 'testing' directory
    for root, _, files in os.walk(dataset_root):
        if 'testing' not in os.path.relpath(root, dataset_root).split(os.sep):
            continue
        for file in files:
            if file.endswith('.jpg'):
                label = os.path.basename(root)
                if label not in label_to_images:
                    label_to_images[label] = []
                label_to_images[label].append(os.path.join(root, file))

    pairs = []

    # Generate positive pairs
    positive_pairs = []
    for label, image_paths in label_to_images.items():
        if len(image_paths) < 2:
            continue
        for i in range(len(image_paths)):
            for j in range(i + 1, len(image_paths)):
                positive_pairs.append((image_paths[i], image_paths[j], 1))

    # Limit the number of positive pairs to match num_pairs
    num_positive_pairs = min(len(positive_pairs), num_pairs // 2)
    pairs.extend(positive_pairs[:num_positive_pairs])

    # Generate negative pairs
    labels = list(label_to_images.keys())
    negative_label_combinations = list(permutations(labels, 2))
    random.shuffle(negative_label_combinations)

    # Add negative pairs to reach num_pairs
    while len(pairs) < num_pairs:
        for label1, label2 in negative_label_combinations:
            if len(pairs) >= num_pairs:
                break
            img1 = random.choice(label_to_images[label1])
            img2 = random.choice(label_to_images[label2])
            pairs.append((img1, img2, 0))

    # Save pairs to file
    with open(pair_file, 'w') as f:
        for img1, img2, label in pairs:
            f.write(f"{img1} {img2} {label}\n")

    print(f"Generated {len(pairs)} pairs, saved to {pair_file}")

def lfw_test_with_features(model, pair_list, feature_save_path=None, tsne_save_path=None, use_nn=False):
    """
    Perform LFW test and optionally save features and t-SNE visualization.
    
    Args:
        model: The neural network model
        pair_list (str): Path to the file containing pairs to test
        feature_save_path (str, optional): Path to save extracted features
        tsne_save_path (str, optional): Path to save t-SNE visualization
        use_nn (bool): If True, use nearest neighbor metric instead of cosine similarity
    
    Returns:
        float: Accuracy score
    """
    s = time.time()
    features = get_features_from_images(model, pair_list, feature_save_path)
    print(f"Features extracted for {len(features)} images")
    t = time.time() - s
    print('Total time is {}, average time is {}'.format(t, t / len(features)))

    if tsne_save_path:
        visualize_with_umap(features, save_path=tsne_save_path, num_people=68)

    if use_nn:
        acc, th = test_performance_with_nn(features, pair_list)
    else:
        acc, th = test_performance(features, pair_list)

    print('LFW face verification accuracy: ', acc, 'threshold: ', th)
    return acc

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a face recognition model on LFW")
    parser.add_argument('--model', type=str, required=True, help="Choose vir or vit")
    parser.add_argument('--test_model_path', type=str, required=True, help="Path to the pretrained model (.pth file)")
    parser.add_argument('--dataset_root', type=str, required=True, help="Root directory for the LFW dataset")
    parser.add_argument('--pair_file', type=str, default="hasil/pair.txt", help="File to save the generated pair list")
    parser.add_argument('--tsne_save_path', type=str, default="hasil/tsne.jpg", help="File to save tsne_plot")
    parser.add_argument('--feature_save_path', type=str, default="hasil/feature.npy", help="Path to save extracted features")
    parser.add_argument('--use_nn', action='store_true', help="Use nearest neighbor metric instead of cosine similarity")

    args = parser.parse_args()

    # Initialize model
    if args.model == "vit":
      from reformer.vit_pytorch import ViT
      model = ViT(
              image_size = 224,
              patch_size = 8,
              num_classes = 68,
              dim = 256,
              depth = 12,
              heads = 8,
              mlp_dim = 2048,
              dropout = 0.1,
              emb_dropout = 0.1
          )
    
    if args.model == "vir":
      from reformer.vir_pytorch import ViR
      model = ViR(
            img_size=224,
            patch_size=8,
            in_channels=3,  # Corrected from 1000 to 3 (input channels for RGB images)
            num_classes=68,
            dim=256,
            depth=12,
            heads=8,
            bucket_size=5,
            n_hashes=1,
            ff_mult=4,  # Added missing argument
            lsh_dropout=0.1,  # Corrected from dropout to lsh_dropout
            ff_dropout=0.1,  # Added missing argument
            emb_dropout=0.1,
            use_rezero=False  # Added missing argument
          )
    
    # Load pretrained weights and move model to GPU
    load_model(model, args.test_model_path)
    model.to(torch.device("cuda"))
    

    # Remove black images from the dataset
    #remove_black_images(args.dataset_root)
    
    # Generate pair list for testing
    #generate_lfw_pair_list(args.dataset_root, args.pair_file, num_pairs=6000)

    # Perform LFW test
    lfw_test_with_features(
        model,
        args.pair_file,
        args.feature_save_path,
        args.tsne_save_path,
        args.use_nn
    )

if __name__ == '__main__':
    main()