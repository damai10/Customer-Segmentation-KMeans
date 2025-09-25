# Customer Segmentation with KMeans Clustering  

## 📌 Project Overview  
This project applies **K-Means Clustering** to segment customers based on their **Age, Annual Income, and Spending Score**.  
The dataset used is **Mall Customers Dataset**, which contains demographic and spending information of mall customers.  

Customer segmentation helps businesses understand their customers better and design targeted marketing strategies.  

---

## 📊 Dataset  
- **File**: `Mall_Customers.csv`  
- **Features**:
  - `CustomerID`
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

---

## 🛠️ Technologies Used  
- Python 3  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

---

## 🚀 Steps in the Project  
1. Data loading and exploration  
2. Exploratory Data Analysis (EDA)  
3. Feature selection (`Age`, `Annual Income`, `Spending Score`)  
4. Finding the optimal number of clusters using **Elbow Method**  
5. Training the **K-Means Model**  
6. Visualizing the clusters and centroids  

---

## 📈 Results  
- The optimal number of clusters was found to be **5**.  
- Customers were segmented into **5 groups** based on their spending behavior and income.  
- These insights can help businesses in targeted promotions and better decision-making.  

---

## ▶️ How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/customer-segmentation-kmeans.git
   cd customer-segmentation-kmeans
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script:  
   ```bash
   jupyter notebook KMeans_Clustering.ipynb
   ```
   or  
   ```bash
   python K-Means\ Clustering.py
   ```

---

## 📌 Future Improvements  
- Try other clustering algorithms (DBSCAN, Hierarchical Clustering).  
- Use PCA for dimensionality reduction and better visualization.  
- Deploy as a web app using Flask/Streamlit.  

---

✍️ Developed by **Durga Damai** as part of **CodeClause Internship**.  
