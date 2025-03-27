import argparse


from KNearestNeighbours import KNearestNeighbours


def main():
    parser = argparse.ArgumentParser(description="K-NN algo")

    parser.add_argument("k", type=int, help="Amount of neighbours (k)")
    parser.add_argument("train_file", type=str, help="Train set file .csv")
    parser.add_argument("test_file", type=str, help="Test set file .csv")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    knn = KNearestNeighbours(args.k, args.train_file, args.test_file)

    if args.interactive:
        knn.interactive_mode()
    else:
        accuracy = knn.score()
        print(f"\nAccuracy of model: {accuracy:.2%}")
        knn.print_predictions()
        knn.plot_accuracy_vs_k(105)

if __name__ == "__main__":
    main()

# TRAINSET = "train.csv"
# TESTSET = "test.csv"
#
#
# knn = KNearestNeighbours(5, TRAINSET, TESTSET)
#
# # print(*knn.train_set, sep='\n')
# # print(*knn.test_set, sep='\n')
#
# accuracy = knn.score()
# knn.print_predictions()
# print(f"Accuracy of model is:{accuracy:.2%}")
# knn.plot_accuracy_vs_k(20)
#
# knn.interactive_mode()
