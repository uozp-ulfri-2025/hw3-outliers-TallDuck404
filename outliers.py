from math import isnan
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from silhouette import *
from torchvision.models import SqueezeNet1_1_Weights
NAN = float("nan")
model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def embed(fn):
    """ Embed the given image with SqueezeNet 1.1.

    Consult https://pytorch.org/hub/pytorch_vision_squeezenet/

    The above link also uses softmax as the final transformation;
    avoid that final step. Convert the output tensor into a numpy
    array and return it.
    """
    
    input_image = Image.open(fn)
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    return np.array(output[0])
    raise NotImplementedError()


def read_data(path):
    """
    Preberi vse slike v podani poti, jih pretvori v "embedding" in vrni rezultat
    kot slovar, kjer so ključi imena slik, vrednosti pa pripadajoči vektorji.
    """
    ans = {}
    for i in os.listdir(path):
        if os.path.isdir(path + '/' + i):
            ans.update(read_data(path + '/' + i))
            continue
        if i.endswith('.jpg') or i.endswith('.png'):
            #print(i)
            picture = embed(path +'/' + i)
            ans[path +'/' + i] = picture
    return ans
    raise NotImplementedError()


def euclidean_dist(r1, r2):
    return np.linalg.norm(r1-r2)
    raise NotImplementedError()


def cosine_dist(d1, d2):
    """
    Vrni razdaljo med vektorjema d1 in d2, ki je smiselno
    pridobljena iz kosinusne podobnosti.
    """
    cos_similar = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
    return 1 - cos_similar
    raise NotImplementedError()


#def silhouette(el, clusters, data, distance_fn=euclidean_dist):
    """
    Za element el ob podanih podatkih data (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data) vrni silhueto za element el.
    """
    raise NotImplementedError()


#def silhouette_average(data, clusters, distance_fn=euclidean_dist):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    raise NotImplementedError()


def group_by_dir(names):
    """ Generiraj skupine iz direktorijev, v katerih se nahajajo slike """
    groups = {}
    for name in names:
        dir_name = os.path.dirname(name)
        if dir_name not in groups:
            groups[dir_name] = []
        groups[dir_name].append(name)
    return list(groups.values())


def order_by_decreasing_silhouette(data, clusters):
    everything = all_elements(clusters)
    ans = {}
    for i in everything:
        ans[i] = silhouette(i, clusters, data, cosine_dist)
    return sorted(ans, key=ans.get, reverse=True)
    raise NotImplementedError()
def to_numpy(idict):
    od = {}
    for n, c in idict.items():
        od[n] = np.array(c)
    return od


if __name__ == "__main__":
    # dataS2 = {"X": [1, 1],
    #       "Y": [0.9, 1],
    #       "Z": [1, 0],
    #       "Z1": [0.8, 0]}
    # dataS2 = to_numpy(dataS2)
    # order = order_by_decreasing_silhouette(dataS2, [["X", "Y"], ["Z", "Z1"]])
    # print(order)
    data = read_data("traffic-signs")
    clusters = group_by_dir(data.keys())
    ordered = order_by_decreasing_silhouette(data, clusters)
    atypical = list(reversed(ordered[-3:]))
    for n in atypical:
        print(n)
    print("UNUSUAL SIGNS", atypical)
    # data = read_data("traffic-signs")
    # clusters = group_by_dir(data.keys())
    # # print(clusters)
    # ordered = order_by_decreasing_silhouette(data, clusters)
    # print("ATYPICAL TRAFFIC SIGNS")
    # for o in ordered[-3:]:
    #     print(o)
