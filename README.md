# Customer Segmentation

A simple Data Science project that segments shopping mall customers into different groups based on their behavior using clustering algorithms (Unsupervised Learning).

## Project Idea

The dataset contains customer information such as age, gender, annual income, and spending score.

Unlike supervised learning problems, there are no predefined labels indicating which segment each customer belongs to. Instead, the model discovers groups of customers with similar characteristics and behaviors on its own.

This is the key difference between this project and tasks like fraud detection, where the correct label for each record is already known (Supervised Learning).

## Project Structure

```
customer-segmentation/
├── data/
│   ├── raw/                       # Place Mall_Customers.csv here
│   └── README.md
├── src/
│   ├── data/
│   │   └── load_data.py           # Data loading
│   ├── features/
│   │   └── preprocessing.py       # Data preprocessing & feature scaling
│   ├── models/
│   │   └── clustering.py          # KMeans, DBSCAN, evaluation & model saving
│   ├── visualization/
│   │   └── eda_plots.py           # Exploratory analysis & cluster visualization
│   └── utils/
│       └── config.py              # Paths and project constants
├── models/                        # Saved models (.pkl)
├── reports/figures/               # Saved visualizations
├── main.py                        # Runs the complete pipeline
└── requirements.txt
```

## How to Run

1. Download the dataset from Kaggle (see `data/README.md`) and place it in:
   ```
   data/raw/Mall_Customers.csv
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the project:
   ```
   python main.py
   ```

## Models Used

The project uses clustering algorithms from the **scikit-learn** library (not implemented from scratch):

- **KMeans** – Partitions the data into a predefined number of clusters (**K**). The **Elbow Method** is used to help determine an appropriate value for K.
- **DBSCAN** – Automatically detects clusters based on data density without requiring the number of clusters in advance. It can also identify **outliers** that do not belong to any cluster.

## Evaluation

Since this is an **Unsupervised Learning** problem and no ground truth labels are available, the clustering performance is evaluated using the **Silhouette Score**.

The Silhouette Score ranges from **-1 to 1**:
- A value close to **1** indicates well-defined and compact clusters.
- A value around **0** indicates overlapping clusters.
- A value close to **-1** suggests that many samples may have been assigned to the wrong cluster.