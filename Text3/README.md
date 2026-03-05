## Conclusion

### Missing Value Handling

In this project, different methods were tested to handle missing values in the dataset. For numerical columns, the **median** method worked better because the dataset contains some extreme values (outliers). Using the mean could shift the values, while the median is more stable in such cases.  

For categorical columns, the **mode** method was used since it replaces missing values with the most frequently occurring category.

---

### Categorical Encoding

Different encoding techniques were applied depending on the type of categorical feature.

- **One-Hot Encoding** was used for nominal features like `MSZoning` where there is no natural order.
- **Ordinal Encoding** was applied to ordered quality features such as `ExterQual`, where values have a natural ranking.
- **Target Encoding** helped in handling high-cardinality features like `Neighborhood`, as it captures the relationship with the target variable.
- **Label Encoding** was also applied in some cases and is generally suitable for tree-based models.

---

### Feature Scaling

Feature scaling was applied to make sure all numerical features are on a similar scale.

- **Standardization (Z-score scaling)** worked well for normally distributed features.
- **Min-Max Scaling** is useful when models are sensitive to feature ranges.
- Other scaling techniques like **MaxAbs Scaling** and **Normalization** were also explored to understand their effect on the dataset.

---

### Overall Observation

Data preprocessing plays a very important role in building a good machine learning model. Proper handling of missing values, correct encoding of categorical variables, and appropriate feature scaling help improve model performance and make the dataset suitable for training machine learning algorithms.