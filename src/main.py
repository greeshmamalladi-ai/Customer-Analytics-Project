import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("../data/SampleSuperstore.csv")

# =====================================================
# DATA CLEANING
# =====================================================

df = df.drop_duplicates()

# =====================================================
# BUSINESS QUESTION 1: TOTAL SALES
# =====================================================

total_sales = df["Sales"].sum()
print(f"Total Sales: ${total_sales:.2f}")

# =====================================================
# BUSINESS QUESTION 2: TOTAL PROFIT
# =====================================================

total_profit = df["Profit"].sum()
print(f"Total Profit: ${total_profit:.2f}")

# =====================================================
# BUSINESS QUESTION 3: SALES BY CATEGORY
# =====================================================

category_sales = df.groupby("Category")["Sales"].sum()

print("\nSales by Category:")
print(category_sales)

highest_category = category_sales.idxmax()

print(f"\nCategory with Highest Sales: {highest_category}")

# =====================================================
# BUSINESS QUESTION 4: PROFIT BY REGION
# =====================================================

region_profit = df.groupby("Region")["Profit"].sum()

print("\nProfit by Region:")
print(region_profit)

highest_region = region_profit.idxmax()

print(f"\nRegion with Highest Profit: {highest_region}")

# =====================================================
# BUSINESS QUESTION 5: SALES BY STATE
# =====================================================

state_sales = df.groupby("State")["Sales"].sum()

highest_state = state_sales.idxmax()

print(f"\nState with Highest Sales: {highest_state}")

# =====================================================
# BUSINESS QUESTION 6: TOP 10 STATES
# =====================================================

top_states = state_sales.sort_values(ascending=False).head(10)

print("\nTop 10 States by Sales:")
print(top_states)

# =====================================================
# BUSINESS QUESTION 7: SALES BY SUB-CATEGORY
# =====================================================

subcategory_sales = df.groupby("Sub-Category")["Sales"].sum()

print("\nSales by Sub-Category:")
print(subcategory_sales)

highest_subcategory = subcategory_sales.idxmax()

print(f"\nSub-Category with Highest Sales: {highest_subcategory}")

# =====================================================
# BUSINESS QUESTION 8: AVERAGE SALES BY REGION
# =====================================================

average_sales = df.groupby("Region")["Sales"].mean()

print("\nAverage Sales by Region:")
print(average_sales)

# =====================================================
# BUSINESS QUESTION 9: SALES BY CUSTOMER SEGMENT
# =====================================================

segment_sales = df.groupby("Segment")["Sales"].sum()

print("\nSales by Customer Segment:")
print(segment_sales)

highest_segment = segment_sales.idxmax()

print(f"\nHighest Sales Segment: {highest_segment}")

# =====================================================
# BUSINESS QUESTION 10: SHIP MODE ANALYSIS
# =====================================================

ship_mode = df["Ship Mode"].value_counts()

print("\nShip Mode Usage:")
print(ship_mode)

most_used_ship_mode = ship_mode.idxmax()

print(f"\nMost Used Ship Mode: {most_used_ship_mode}")

# =====================================================
# BUSINESS QUESTION 11: PROFIT BY CATEGORY
# =====================================================

category_profit = df.groupby("Category")["Profit"].sum()

print("\nProfit by Category:")
print(category_profit)

highest_profit_category = category_profit.idxmax()

print(f"\nCategory with Highest Profit: {highest_profit_category}")

# =====================================================
# BUSINESS QUESTION 12: TOP 10 CITIES BY SALES
# =====================================================

city_sales = df.groupby("City")["Sales"].sum()

top_city = city_sales.sort_values(ascending=False).head(10)

print("\nTop 10 Cities by Sales:")
print(top_city)

# =====================================================
# VISUALIZATIONS
# =====================================================

# Chart 1 - Sales by Category
category_sales.sort_values(ascending=False).plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.savefig("../images/sales_by_category.png")
plt.show()
plt.close()

# Chart 2 - Sales Distribution by Category
category_sales.plot(kind="pie", autopct="%1.1f%%")
plt.title("Sales Distribution by Category")
plt.ylabel("")
plt.savefig("../images/sales_distribution.png")
plt.show()
plt.close()

# Chart 3 - Profit by Region
region_profit.sort_values(ascending=False).plot(kind="bar")
plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")
plt.savefig("../images/profit_by_region.png")
plt.show()
plt.close()

# Chart 4 - Top 10 States by Sales
top_states.plot(kind="bar")
plt.title("Top 10 States by Sales")
plt.xlabel("State")
plt.ylabel("Sales")
plt.savefig("../images/top10_states.png")
plt.show()
plt.close()

# Chart 5 - Top 10 Sub-Categories by Sales
top_subcategory = subcategory_sales.sort_values(ascending=False).head(10)

top_subcategory.plot(kind="bar")
plt.title("Top 10 Sub-Categories by Sales")
plt.xlabel("Sub-Category")
plt.ylabel("Sales")
plt.savefig("../images/top10_subcategories.png")
plt.show()
plt.close()

# Chart 6 - Average Sales by Region
average_sales.sort_values(ascending=False).plot(kind="bar")
plt.title("Average Sales by Region")
plt.xlabel("Region")
plt.ylabel("Average Sales")
plt.savefig("../images/average_sales_region.png")
plt.show()
plt.close()

# Chart 7 - Sales by Customer Segment
segment_sales.sort_values(ascending=False).plot(kind="bar")
plt.title("Sales by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Sales")
plt.savefig("../images/customer_segment.png")
plt.show()
plt.close()

# Chart 8 - Ship Mode Usage
ship_mode.plot(kind="bar")
plt.title("Most Used Ship Mode")
plt.xlabel("Ship Mode")
plt.ylabel("Number of Orders")
plt.savefig("../images/ship_mode.png")
plt.show()
plt.close()

# Chart 9 - Profit by Category
category_profit.sort_values(ascending=False).plot(kind="bar")
plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.savefig("../images/profit_by_category.png")
plt.show()
plt.close()

# Chart 10 - Top 10 Cities by Sales
top_city.plot(kind="bar")
plt.title("Top 10 Cities by Sales")
plt.xlabel("City")
plt.ylabel("Sales")
plt.savefig("../images/top10_cities.png")
plt.show()
plt.close()

# =====================================================
# MACHINE LEARNING - LINEAR REGRESSION
# =====================================================

# Features (Input)
X = df[["Quantity", "Discount", "Profit"]]

# Target (Output)
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print(f"\nMean Absolute Error: {mae:.2f}")

joblib.dump(model, "../models/sales_prediction_model.pkl")

print("Machine Learning model saved successfully!")