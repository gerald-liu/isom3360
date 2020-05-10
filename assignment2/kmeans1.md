# Assignment 2 - K-Means Clustering

The initial seeds are (2,10), (5,8), (1,2).

Computer the distance between each point and seed (dist{} refers to the distance to seed{}):

|      | coordinate |  dist1 |  dist2 |  dist3 |    y |
| :--- | :--------- | -----: | -----: | -----: | ---: |
| A1   | (2, 10)    |      0 | 3.6056 | 8.0623 |    1 |
| A2   | (2, 5)     |      5 | 4.2426 | 3.1623 |    3 |
| A3   | (8, 4)     | 8.4853 |      5 | 7.2801 |    2 |
| A4   | (5, 8)     | 3.6056 |      0 | 7.2111 |    2 |
| A5   | (7, 5)     | 7.0711 | 3.6056 | 6.7082 |    2 |
| A6   | (6, 4)     | 7.2111 | 4.1231 | 5.3852 |    2 |
| A7   | (1, 2)     | 8.0623 | 7.2111 |      0 |    3 |
| A8   | (4, 9)     | 2.2361 | 1.4142 | 7.6158 |    2 |

Assign each point to the closest centroid:

| Centroid of the cluster | Points in the cluster |
| ----------------------- | --------------------- |
| (2, 10)                 | A1                    |
| (5, 8)                  | A3, A4, A5, A6, A8    |
| (1, 2)                  | A2, A7                |

In each cluster, average the points to get the new centroid:

| Points in the cluster             | Centroid of the cluster |
| --------------------------------- | ----------------------- |
| (2, 10)                           | (2, 10)                 |
| (8,4), (5,8), (7,5), (6,4), (4,9) | (6, 6)                  |
| (2,5), (1,2)                      | (1.5, 3.5)              |
