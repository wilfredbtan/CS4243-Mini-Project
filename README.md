# CS4243 Mini-project

## Setting up (on local machine)

- Download the dataset (the folder `cs4243_smallest`) from [here](https://drive.google.com/drive/folders/1pCEBqzQDTJ3PlgdIRBY65jOugJ4xpFi6?usp=sharing)
- Put the folder `cs4243_smallest` in the root directory of the project.
- Get the `dataset.csv` file with the file paths and their labels:

  ```zsh
    chmod +x ./initData.sh
    ./initData.sh
  ```

- Get `train_label.csv` and `test_label.csv` files:

  ```zsh
    chmod +x ./splitTrainTest.sh
    ./splitTrainTest.sh
  ```

### Setting up (on google colab)

- Open file on Google Colab
- Create a shortcut on the drive to `cs4243_smallest` from [here](https://drive.google.com/drive/folders/1pCEBqzQDTJ3PlgdIRBY65jOugJ4xpFi6?usp=sharing)
- Edit all file paths (marked with `## CHANGE ##`) to fit ur  project environment
- If running code for the first time, run the bash command to get the training and testing datasets (marked with `## COMMENT OUT AFTER RUNNING ONCE ##`)
- If using an already trained model with a corresponding pth file, comment out the training function (marked with `## COMMENT OUT IF NOT TRAINING ##`)
