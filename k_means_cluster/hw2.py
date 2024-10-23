import re
import numpy as np
from tabulate import tabulate
preprocessed_text = ""

with open("gdnhealthcare.txt", "r", encoding='latin1') as file:
    for line in file:
        #Remove id and timestamp
        new_line = line.find('|', line.find('|') + 1)
        no_id_time = line[new_line:]
        no_id_time = no_id_time.replace('|',"")
        #remove url
        no_url = re.sub(r'https?://\S+', '', no_id_time)
        #remove symbol
        no_symbol = re.sub(r'@\w+\b', '', no_url)
        #remove hashtag
        no_hashtag = no_symbol.replace('#','')
        #lowercase
        res = no_hashtag.lower()
        preprocessed_text += res

with open("preprocessed.txt", "w", encoding='latin1') as new_file:
    new_file.write(preprocessed_text)


tweets = []
with open("preprocessed.txt", "r", encoding='latin1') as file:
    for line in file:
        tweets.append(line.strip())


def jaccard_distance(tweetA, tweetB):
    setA = set(tweetA.split())
    setB = set(tweetB.split())
    intersection = len(setA.intersection(setB))
    union = len(setA.union(setB))
    if union != 0:
        dist = 1 - (intersection / union)
    else:
        0
    return dist
def initialize_centroids(tweets, k):
    #Used random
    indices = np.random.choice(len(tweets), k, replace=False)
    centroids = np.array([tweets[i] for i in indices])
    return centroids

def assign_to_clusters(tweets, centroids):
    #Assign tweets to clusters
    clusters = {}
    for tweet in tweets:
        distances = [jaccard_distance(tweet, centroid) for centroid in centroids]
        closest = np.argmin(distances)
        if closest not in clusters:
            clusters[closest] = []
        clusters[closest].append(tweet)
    return clusters

def update_centroids(clusters):
    #Update centroids
    new_centroids = []
    for cluster_tweets in clusters.values():
        counts = {tweet: cluster_tweets.count(tweet) for tweet in set(cluster_tweets)}
        most_common, _ = max(counts.items(), key=lambda x: x[1])
        new_centroids.append(most_common)
    return new_centroids

def calculate_sse(clusters, centroids):
    sse = 0
    for i, centroid in enumerate(centroids):
        for tweet in clusters[i]:
            sse += (jaccard_distance(tweet, centroid)) ** 2
    return sse

def k_means_cluster(tweets, k, iterations=100):
    centroids = initialize_centroids(tweets, k)
    for _ in range(iterations):
        clusters = assign_to_clusters(tweets, centroids)
        new_centroids = update_centroids(clusters)
        #Check for convergence
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    sse = calculate_sse(clusters, centroids)
    return clusters, sse

table = []
for k in range(1, 6):
    clusters, sse = k_means_cluster(tweets, k)
    cluster_sizes = [len(cluster_tweets) for cluster_tweets in clusters.values()]
    table.append([k, sse] + cluster_sizes)

headers = ["K", "SSE"] + [f"Cluster {i+1} Size" for i in range(max(map(len, table)))]

print(tabulate(table, headers=headers, tablefmt="grid"))


