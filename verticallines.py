""" bubble
import random
import matplotlib.pyplot as plt

# Generate 40 random points
random_points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]

# Define a threshold for horizontal alignment
horizontal_threshold = 0.5  # Adjust this value based on your requirement

# Plot all points
plt.figure(figsize=(8, 6))
for point in random_points:
    plt.scatter(point[0], point[1], color='blue')

# Connect dots with lines based on the horizontal distance threshold
for i in range(len(random_points)):
    for j in range(i + 1, len(random_points)):
        x1, y1 = random_points[i]
        x2, y2 = random_points[j]
        if abs(x2 - x1) < horizontal_threshold:
            plt.plot([x1, x2], [y1, y2], color='green')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points with Lines Connecting Close Horizontal Alignment (±0.5)')
plt.grid(True)
plt.show()
"""

""" sorted
import random
import matplotlib.pyplot as plt

# Generate 40 random points
random_points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]

# Define a threshold for horizontal alignment
horizontal_threshold = 2  # Adjust this value based on your requirement

# Sort the points based on x-coordinate
random_points.sort()

# Plot all points
plt.figure(figsize=(8, 6))
for point in random_points:
    plt.scatter(point[0], point[1], color='blue')

# Connect dots with lines based on the horizontal distance threshold
for i in range(len(random_points)):
    for j in range(i + 1, len(random_points)):
        x1, y1 = random_points[i]
        x2, y2 = random_points[j]
        
        # If the horizontal distance exceeds the threshold, break the inner loop
        if abs(x2 - x1) > horizontal_threshold:
            break
        
        # If the horizontal distance is within the threshold, draw a line
        if abs(x2 - x1) < horizontal_threshold:
            plt.plot([x1, x2], [y1, y2], color='green')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points with Lines Connecting Close Horizontal Alignment (±0.5)')
plt.grid(True)
plt.show()
"""

""" sweep line
import random
import matplotlib.pyplot as plt

# Generate 40 random points
random_points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]

# Define a threshold for horizontal alignment
horizontal_threshold = 2  # Adjust this value based on your requirement

# Sort the points based on x-coordinate
random_points.sort()

# Plot all points
plt.figure(figsize=(8, 6))
for point in random_points:
    plt.scatter(point[0], point[1], color='blue')

# Sweep line algorithm to connect dots with lines based on the horizontal distance threshold
for i in range(len(random_points)):
    x1, y1 = random_points[i]
    for j in range(i + 1, len(random_points)):
        x2, y2 = random_points[j]
        
        # If the horizontal distance exceeds the threshold, break the inner loop
        if abs(x2 - x1) > horizontal_threshold:
            break
        
        # If the horizontal distance is within the threshold, draw a line
        if abs(x2 - x1) < horizontal_threshold:
            plt.plot([x1, x2], [y1, y2], color='green')
            
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points with Lines Connecting Close Horizontal Alignment (±0.5)')
plt.grid(True)
plt.show()
"""

"""
#sweep with bucket
import random
import matplotlib.pyplot as plt

# Generate 40 random points
random_points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(40)]

# Define a threshold for horizontal alignment
horizontal_threshold = 2  # Adjust this value based on your requirement

# Sort the points based on x-coordinate
random_points.sort()

# Define the number of buckets and bucket width
num_buckets = 10
bucket_width = 100 / num_buckets

# Initialize buckets
buckets = [[] for _ in range(num_buckets)]

# Assign points to buckets
for point in random_points:
    bucket_index = min(int(point[0] / bucket_width), num_buckets - 1)
    buckets[bucket_index].append(point)

# Plot all points
plt.figure(figsize=(8, 6))
for point in random_points:
    plt.scatter(point[0], point[1], color='blue')

# Sweep line with bucketing algorithm to connect dots with lines based on the horizontal distance threshold
for bucket in buckets:
    bucket.sort(key=lambda x: x[1])  # Sort points in each bucket by y-coordinate
    for i in range(len(bucket)):
        x1, y1 = bucket[i]
        for j in range(i + 1, len(bucket)):
            x2, y2 = bucket[j]
            
            # If the horizontal distance exceeds the threshold, break the inner loop
            if abs(x2 - x1) > horizontal_threshold:
                break
            
            # If the horizontal distance is within the threshold, draw a line
            if abs(x2 - x1) < horizontal_threshold:
                plt.plot([x1, x2], [y1, y2], color='green')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points with Lines Connecting Close Horizontal Alignment (±0.5)')
plt.grid(True)
plt.show()
"""

import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Function to connect points within a bucket
def connect_points(bucket):
    lines = []
    for i in range(len(bucket)):
        x1, y1 = bucket[i]
        for j in range(i + 1, len(bucket)):
            x2, y2 = bucket[j]
            if abs(x2 - x1) < horizontal_threshold:
                lines.append([(x1, y1), (x2, y2)])
    return lines

# Generate 40 random points
random_points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(100)]

# Define a threshold for horizontal alignment
horizontal_threshold = 0.5  # Adjust this value based on your requirement

# Sort the points based on x-coordinate
random_points.sort()

# Define the number of buckets and bucket width
num_buckets = 10
bucket_width = 100 / num_buckets

# Initialize buckets
buckets = [[] for _ in range(num_buckets)]

# Assign points to buckets
for point in random_points:
    bucket_index = min(int(point[0] / bucket_width), num_buckets - 1)
    buckets[bucket_index].append(point)

# Plot all points
plt.figure(figsize=(8, 6))
for point in random_points:
    plt.scatter(point[0], point[1], color='blue')

# Create a pool of worker processes
with Pool() as pool:
    # Parallelize the task of connecting points within each bucket
    lines = pool.map(connect_points, buckets)
    
    # Plot the lines connecting the points
    for bucket_lines in lines:
        for line in bucket_lines:
            plt.plot([point[0] for point in line], [point[1] for point in line], color='green')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points with Lines Connecting Close Horizontal Alignment (±0.5)')
plt.grid(True)
plt.show()
