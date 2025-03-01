from math import sqrt,isnan
NAN = float("nan")

def manhattan_dist(r1, r2):
    """ Arguments r1 and r2 are lists of numbers """
    ans = 0.0
    length = len(r1)
    change = False
    for i in range(len(r1)):
        if(isnan(r1[i]) or isnan(r2[i])):
            length -= 1
            continue
        change = True
        ans += abs(r1[i] - r2[i])
    if not change:
        return NAN
    ans = ans*len(r1)/length
    return ans


def euclidean_dist(r1, r2):
    ans = 0.0
    length = len(r1)
    change = False
    for i in range(len(r1)):
        if(isnan(r1[i]) or isnan(r2[i])):
            length -= 1
            continue
        change = True
        ans += (r1[i] - r2[i])**2
    if not change:
        return NAN
    ans = ans*len(r1)/length
    return sqrt(ans)


def single_linkage(c1, c2, distance_fn):
    """ Arguments c1 and c2 are lists of lists of numbers
    (lists of input vectors or rows).
    Argument distance_fn is a function that can compute
    a distance between two vectors (like manhattan_dist)."""
    dist = float('inf')
    change = False
    for i in c1:
        for j in c2:
            if distance_fn(i, j) < dist:
                change = True
                dist = distance_fn(i, j)
    if not change:
        return NAN
    return dist



def complete_linkage(c1, c2, distance_fn):
    dist = -1.0
    change = False
    for i in c1:
        for j in c2:
            if distance_fn(i, j) > dist:
                change = True
                dist = distance_fn(i, j)
    if not change:
        return NAN
    return dist


def average_linkage(c1, c2, distance_fn):
    dist = 0.0
    length = (len(c1)*len(c2))
    for i in c1:
        for j in c2:
                if isnan(distance_fn(i, j)):
                    length -= 1
                else:
                    dist += distance_fn(i, j)
    if length == 0:
        return NAN
    return dist/length

def cluster_to_values(data, cluster):
        ans = []
        for i in cluster:
            if isinstance(i, float) or isinstance(i, int):
                continue
            elif isinstance(i, list):
                ans = ans + cluster_to_values(data, i)
            else:
                ans.append(data.get(i))
        return ans

class HierarchicalClustering:

    def __init__(self, cluster_dist, return_distances=False):
        # the function that measures distances clusters (lists of data vectors)
        self.cluster_dist = cluster_dist

        # if the results of run() also needs to include distances;
        # if true, each joined pair in also described by a distance.
        self.return_distances = return_distances
    

    def closest_clusters(self, data, clusters):
        """
        Return the closest pair of clusters and their distance.
        """
        first = None
        second = None
        distance = float('inf')
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
               # print(self.cluster_to_values(data,clusters[i]), self.cluster_to_values(data,clusters[j]))
                if self.cluster_dist(cluster_to_values(data,clusters[i]), cluster_to_values(data,clusters[j])) < distance:
                    distance = self.cluster_dist(cluster_to_values(data,clusters[i]), cluster_to_values(data,clusters[j]))
                    first = clusters[i]
                    second = clusters[j]
        return first, second, distance

    def run(self, data):
        """
        Performs hierarchical clustering until there is only a single cluster left
        and return a recursive structure of clusters.
        """

        # clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into lists like
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        clusters = [[name] for name in data.keys()]

        while len(clusters) >= 2:
            first, second, distance = self.closest_clusters(data, clusters)
            # update the "clusters" variable
            if(self.return_distances):
                temp = [first,  second, distance]
            else:
                temp = [first,  second]
            clusters.remove(first)
            clusters.remove(second)
            clusters.append(temp)
        return clusters


if __name__ == "__main__":

    data = {"a": [1, 2],
            "b": [2, 3],
            "c": [5, 5]}

    def average_linkage_w_manhattan(c1, c2):
        return average_linkage(c1, c2, manhattan_dist)
    
    small_data = {"a": [1, 2],
                  "b": [NAN, 1],
                  "c": [5, NAN],
                  "d": [NAN, 1],
                  "e": [12, 3]}
    
    hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan)
    # clusters = hc.run(data)
    # print(clusters)  # [[['c'], [['a'], ['b']]]] (or equivalent)
    
    print(complete_linkage([[NAN], [1]], [[NAN]], manhattan_dist))
    # hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan,
    #                             return_distances=True)
    # clusters = hc.run(data)
    # print(clusters)  # [[['c'], [['a'], ['b'], 2.0], 6.0]] (or equivalent)
