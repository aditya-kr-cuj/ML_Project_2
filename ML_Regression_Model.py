import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import messagebox

# Step 1: Prepare the Dataset
data = {
    'Feature': [1, 2, 3, 4, 5],
    'Target': [2.2, 4.1, 6.0, 7.9, 9.8]
}
df = pd.DataFrame(data)

X = df[['Feature']]
Y = df['Target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Step 2: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Implement Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# Step 4: Evaluate the Model
Y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Step 5: Visualize the Regression Line
def show_plot():
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(scaler.transform(X)), color='red', label='Regression Line')
    plt.title('Linear Regression')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 6: GUI for Linear Regression
def predict_target():
    try:
        # Get user input from the entry field
        user_input = float(entry.get())
        
        # Scale the input
        user_input_scaled = scaler.transform([[user_input]])
        
        # Predict the target
        predicted_value = model.predict(user_input_scaled)
        
        # Show the predicted value in a popup
        messagebox.showinfo("Prediction Result", f"The predicted target value for the feature {user_input} is: {predicted_value[0]:.2f}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid numerical value.")

# Create the Tkinter GUI
root = tk.Tk()
root.title("Linear Regression Tool")

# Add a label for the tool
label = tk.Label(root, text="Linear Regression Feature and Target Tool", font=("Arial", 16))
label.pack(pady=10)

# Add a button to show the regression plot
plot_button = tk.Button(root, text="Show Regression Plot", command=show_plot)
plot_button.pack(pady=10)

# Add a label for input
input_label = tk.Label(root, text="Enter a value for the feature:")
input_label.pack(pady=10)

# Add an entry field for user input
entry = tk.Entry(root)
entry.pack(pady=5)

# Add a button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict_target)
predict_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
