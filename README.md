# 🛒 E-commerce Customer Churn Prediction

### 📌 Introduction
In e-commerce, the cost of acquiring a new customer is **5 to 25 times higher** than retaining an existing one. However, many businesses still rely on broad, unfocused marketing strategies that fail to identify customers who are most at risk of churning.

Based on this dataset, approximately **16.8% of customers churn** — about 1 in 6. Left unaddressed, this churn rate can result in:
- 📉 Revenue losses  
- ⬇️ Lower customer lifetime value  
- 💸 Inefficient marketing spend  

Generic loyalty programs or discounts are not cost-effective — they risk **misallocating resources** toward users who never intended to leave.

> **Definition**:  
> - `Churn = 1` → Customer has left the platform  
> - `Churn = 0` → Customer is still active  

> **References**:  
> • [Customer Retention vs Acquisition (Forbes)](https://www.forbes.com/councils/forbesbusinesscouncil/2022/12/12/customer-retention-versus-customer-acquisition)  
> • [Zero Defections – HBR](https://hbr.org/1990/09/zero-defections-quality-comes-to-services)

---

## 📈 Final Model Summary

- **Model**: Tuned LightGBM Classifier  
- **Resampling**: Random Over Sampling (ROS)  
- **Feature Selection**: SelectKBest (f_classif)  
- **Primary Metric**: **F2-Score = 0.901379**

### 🔍 Prediction Breakdown
| Category                       | Count |
|--------------------------------|-------|
| ✅ True Negatives (0 → 0)      | 923   |
| ⚠️ False Positives (0 → 1)     | 13    |
| 🎯 True Positives (1 → 1)      | 170   |
| ❌ False Negatives (1 → 0)     | 20    |

**Interpretation**:
- High success in correctly identifying **non-churners**
- Strong **recall** in spotting actual churners
- Low number of **false positives** and **false negatives**
- Suitable for **targeted retention campaigns** with minimal wasted budget

---

## 👥 Stakeholder Impact

The **Customer Marketing Team** can leverage the model to:
- Proactively engage high-risk customers  
- Minimize churn through early intervention  
- Allocate retention resources more efficiently

---

## 🧰 Tools & Libraries Used
- `Python`
- `Pandas`, `NumPy`
- `Scikit-learn`, `Imbalanced-learn`
- `LightGBM`
- `Matplotlib`, `Seaborn`
---

## 🧑‍🤝‍🧑 Team Members
- **Meriani Alexandra**  
- **Nadame Kristina**
