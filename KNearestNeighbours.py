import math
from collections import Counter

from matplotlib import pyplot as plt

from InvalidInputError import InvalidInputError


class KNearestNeighbours:
    def __init__(self, k, train_set_file, test_set_file):
        self.k = k
        self.train_set = self.load_data(train_set_file)
        self.test_set = self.load_data(test_set_file)
        self.actual_classes = []
        self.predicted_classes = []


    def load_data(self, train_set, skip_header=True):
        dataset = []
        with open(train_set, 'r') as file:
            lines = file.readlines()
            if skip_header:
                lines = lines[1:]

            for line in lines:
                row = line.strip().split(';')
                dataset.append([float(x) if i < len(row) - 1 else x for i, x in enumerate(row)])
        return dataset

    def euclidean_distance(self, vector1, vector2):
        distance = 0
        for i in range(len(vector1) - 1):
            distance += (vector1[i] - vector2[i]) ** 2
        return distance ** 0.5

    def get_neighbours(self, test_row):
        distances = []
        for train_row in self.train_set:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))

        distances.sort(key=lambda x: x[1])
        k_neighbours = min(self.k, len(distances))
        neighbours = [distances[i][0] for i in range(k_neighbours)]
        return neighbours

    def predict(self, test_row):
        neighbours = self.get_neighbours(test_row)
        output_classes = [row[-1] for row in neighbours]
        prediction = Counter(output_classes).most_common(1)[0][0]
        return prediction

    def score(self):
        correct = 0
        self.actual_classes = []
        self.predicted_classes = []

        for test_row in self.test_set:
            actual = test_row[-1]
            predicted = self.predict(test_row)

            self.actual_classes.append(actual)
            self.predicted_classes.append(predicted)

            if predicted == actual:
                correct+=1

        return correct / len(self.test_set)

    def print_predictions(self):
        for i in range(len(self.actual_classes)):
            print(f"{self.actual_classes[i]}     {self.predicted_classes[i]}")


    def interactive_mode(self):
        num_features = len(self.train_set[0]) - 1
        print("Enter vector using ';' as separator or exit for escaping")

        while True:
            user_input = input("Enter vector: ")

            if user_input.lower() == "exit":
                break

            try:
                user_vector = [float(x) for x in user_input.split(";")]

                if len(user_vector) != num_features:
                    raise InvalidInputError(f"Error! Expected {num_features} arguments, received {len(user_vector)}.")


                predicted_class = self.predict(user_vector + ["-"])
                print(f"Predicted class: {predicted_class}\n")

            except ValueError:
                print("Check if attributes are numerical and separated with ';' ")
            except InvalidInputError as e:
                print(f"{e}")




    def plot_accuracy_vs_k(self, max_k):
        ks = range(1, max_k + 1)
        accuracies = []

        for k in ks:
            self.k = k
            accuracy = self.score()
            accuracies.append(accuracy)

        plt.figure(figsize=(8, 5))
        plt.plot(ks, accuracies, marker='o', linestyle='-')
        plt.xlabel("Amount of neghbours (k)")
        plt.ylabel("Accuracy")
        plt.title("Dependence of accuracy to k")
        plt.grid(True)
        plt.show()