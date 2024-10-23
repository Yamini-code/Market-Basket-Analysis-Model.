# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:12:07 2024

@author: reddy
"""

# Let's load the CSV file and display its contents to understand its structure for further processing.
import pandas as pd

# Load the dataset
data = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Preview the dataset
print(data.head())


from mlxtend.preprocessing import TransactionEncoder

# Convert the dataset into a list of transactions
transactions = []
for i in range(0, len(data)):
    transactions.append([str(data.values[i,j]) for j in range(0, 20) if str(data.values[i,j]) != 'nan'])

# Use TransactionEncoder to transform the list into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(df)
from mlxtend.frequent_patterns import apriori, association_rules

# Calculate the frequent itemsets
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Preview the generated rules
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head()
print(rules)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
plt.subplot(3, 1, 1)
plt.bar(range(len(rules)), rules['support'], color='blue')
plt.title('Support of Rules')
plt.xlabel('Rules')
plt.ylabel('Support')

# Plot Confidence
plt.subplot(3, 1, 2)
plt.bar(range(len(rules)), rules['confidence'], color='green')
plt.title('Confidence of Rules')
plt.xlabel('Rules')
plt.ylabel('Confidence')
# Plot Lift
plt.subplot(3, 1, 3)
plt.bar(range(len(rules)), rules['lift'], color='red')
plt.title('Lift of Rules')
plt.xlabel('Rules')
plt.ylabel('Lift')

plt.tight_layout()
plt.show()

