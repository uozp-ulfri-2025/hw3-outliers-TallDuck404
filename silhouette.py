from hc import *
def el_in_cluster(el,cluster):
    if(isinstance(cluster, list)):
       return any(el_in_cluster(el,i) for i in cluster)
    return el == cluster

def sum_distances(el,clusters,data, distance_fn=euclidean_dist):
    values = cluster_to_values(data,clusters)
    el = data.get(el)
    ans = 0.0
    for i in values:
        ans += distance_fn(el,i)
    return ans, len(values)
def a(i,cluster,data, distance_fn=euclidean_dist):
    for j in cluster:
        if el_in_cluster(i,j):
            ans, length = sum_distances(i,j,data,distance_fn)
            if length == 1:
                return NAN
            return ans/(length-1)
    return NAN
def b(i,cluster,data, distance_fn=euclidean_dist):
    ans = float('inf')
    for j in cluster:
        if not el_in_cluster(i,j):
            temp, length = sum_distances(i,j,data,distance_fn)
            ans = min(ans, temp/length)
    return ans
    

def silhouette(el, clusters, data, distance_fn=euclidean_dist):
    """
    Za element el ob podanih podatkih data (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data) vrni silhueto za element el.
    """
    a1 = a(el, clusters, data,distance_fn)
    if isnan(a1):
        return 0
    b1 = b(el, clusters, data,distance_fn)
    return (b1-a1)/max(a1, b1)

def all_elements(clusters):
    ans = []
    for i in clusters:
        if isinstance(i, list):
            ans += all_elements(i)
        else:
            ans.append(i)
    ans = list(set(ans))
    return ans
def silhouette_average(data, clusters, distance_fn=euclidean_dist):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    everything = all_elements(clusters)
    ans = 0.0
    for i in everything:
        ans += silhouette(i, clusters, data,distance_fn)
    return ans/len(everything)
if __name__ == "__main__":
    
    dataS2 = {"X": [1, 1],
          "Y": [0.9, 1],
          "Z": [1, 0],
          "Z1": [0.8, 0]}
    clusters = [["X", "Y"], ["Z", "Z1"]]
    print(all_elements(clusters))
    print(silhouette_average(dataS2, [["X", "Y"], ["Z", "Z1"]]))
