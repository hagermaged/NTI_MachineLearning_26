import pandas as pd
# Load
data = pd.read_csv("Housing.csv")
data.columns = data.columns.str.strip().str.lower()
data['guestroom'] = data['guestroom'].replace({'yes': 1, 'no': 0})

# Exercise 1 
premium = data[(data['guestroom'] == 1) &
               (data['area'] > 3000) &
               (data['airconditioning'] == 'yes')]

ex1 = premium.groupby('furnishingstatus').agg(
    avg_price=('price', 'mean'),
    avg_bedrooms=('bedrooms', 'mean'),
    count=('price', 'size')
)

# Exercise 2
data['energy_load'] = (data['area'] * 0.5) + (data['bedrooms'] * 100) + (data['bathrooms'] * 150)

sub = data[(data['airconditioning'] == 'yes') & (data['hotwaterheating'] == 'yes')]

ex2 = sub.groupby('furnishingstatus').agg({'price': 'mean', 'energy_load': 'mean'})

print("Exercise 1 result:")
print(ex1)
print("\nExercise 2 result:")
print(ex2)