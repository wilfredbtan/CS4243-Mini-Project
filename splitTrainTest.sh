
echo "directory,label" > train_label.csv
echo "directory,label" > test_label.csv
awk -v seed=999 'BEGIN{srand(seed);}(NR>1){if(rand()<0.9) {print $0 >> "train_label.csv"} else {print $0 >> "test_label.csv"}}' dataset.csv