from flask import Flask, render_template, request, redirect, url_for
import math
import time
import random
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def closest_pair_bruteforce(points):
    start_time = time.time()
    min_distance = float('inf')
    n = len(points)
    pair = None
    for i in range(n):
        for j in range(i + 1, n):
            distance = math.dist(points[i], points[j])
            if distance < min_distance:
                min_distance = distance
                pair = (points[i], points[j])
    end_time = time.time()
    execution_time = end_time - start_time
    return pair, min_distance, execution_time

def closest_pair_divide_and_conquer(points):
    def closest_pair_rec(sorted_x, sorted_y):
        if len(sorted_x) <= 3:
            return closest_pair_bruteforce(sorted_x)
        
        mid = len(sorted_x) // 2
        midpoint = sorted_x[mid]
        
        sorted_y_left = list(filter(lambda x: x[0] <= midpoint[0], sorted_y))
        sorted_y_right = list(filter(lambda x: x[0] > midpoint[0], sorted_y))
        
        (p1, q1, dist1, time1) = closest_pair_rec(sorted_x[:mid], sorted_y_left)
        (p2, q2, dist2, time2) = closest_pair_rec(sorted_x[mid:], sorted_y_right)
        
        if dist1 <= dist2:
            min_pair = (p1, q1)
            min_dist = dist1
            min_time = time1
        else:
            min_pair = (p2, q2)
            min_dist = dist2
            min_time = time2
        
        strip = [p for p in sorted_y if abs(p[0] - midpoint[0]) < min_dist]
        min_pair_strip, min_dist_strip, time_strip = closest_in_strip(strip, min_dist)
        
        if min_dist_strip < min_dist:
            return min_pair_strip, min_dist_strip, min_time + time_strip
        else:
            return min_pair, min_dist, min_time

    def closest_in_strip(strip, d):
        min_dist = d
        pair = None
        len_strip = len(strip)
        start_time = time.time()
        for i in range(len_strip):
            for j in range(i + 1, len_strip):
                if (strip[j][1] - strip[i][1]) < min_dist:
                    distance = math.dist(strip[i], strip[j])
                    if distance < min_dist:
                        min_dist = distance
                        pair = (strip[i], strip[j])
        end_time = time.time()
        execution_time = end_time - start_time
        return pair, min_dist, execution_time
    
    start_time = time.time()
    sorted_x = sorted(points, key=lambda x: x[0])
    sorted_y = sorted(points, key=lambda x: x[1])
    min_pair, min_dist, rec_time = closest_pair_rec(sorted_x, sorted_y)
    end_time = time.time()
    total_time = end_time - start_time + rec_time
    
    return min_pair, min_dist, total_time
def closest_pair_genetic(points, population_size=100, generations=1000, mutation_rate=0.01):
    start_time = time.time()
    
    def random_pair():
        return random.sample(points, 2)
    
    def fitness(pair):
        return -math.dist(pair[0], pair[1])
    
    def mutate(pair):
        if random.random() < mutation_rate:
            return random_pair()
        return pair
    
    def crossover(parent1, parent2):
        return random.choice([parent1, parent2])
    
    population = [random_pair() for _ in range(population_size)]
    for _ in range(generations):
        population = sorted(population, key=fitness)
        next_population = population[:10]  # Elitism: keep top 10 pairs
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(population[:50], 2)
            child = mutate(crossover(parent1, parent2))
            next_population.append(child)
        population = next_population
    
    best_pair = population[0]
    min_distance = -fitness(best_pair)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return best_pair, min_distance, execution_time

def plot_points(points, closest_pair):
    plt.figure()
    x, y = zip(*points)
    plt.scatter(x, y, label='Points')
    plt.plot([closest_pair[0][0], closest_pair[1][0]], [closest_pair[0][1], closest_pair[1][1]], 'r-', label='Closest Pair')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Closest Pair of Points')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            points = request.form['points']
            points = eval(points)
            
            # Validasi input sebagai list of tuples 3D
            if isinstance(points, list) and all(isinstance(t, tuple) and len(t) == 3 for t in points):
                algorithm = request.form['algorithm']
                if algorithm == 'bruteforce':
                    result = closest_pair_bruteforce(points)
                elif algorithm == 'divide_and_conquer':
                    result = closest_pair_divide_and_conquer(points)
                elif algorithm == 'genetic':
                    result = closest_pair_genetic(points)
                plot_url = plot_points(points, result[0])
                return render_template('index.html', result=result, plot_url=plot_url)
            else:
                error_message = "Invalid input format. Please enter a list of tuples like [(x1, y1, z1), (x2, y2, z2), ...]."
                return render_template('index.html', error_message=error_message)
        except Exception as e:
            error_message = str(e)
            return render_template('index.html', error_message=error_message)
    
    return render_template('index.html')

def plot_points(points, closest_pair):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = zip(*points)
    ax.scatter(x, y, z, label='Points')
    
    cx, cy, cz = zip(*closest_pair)
    ax.plot(cx, cy, cz, 'r-', label='Closest Pair')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.title('Closest Pair of Points in 3D')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
