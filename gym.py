import matplotlib.pyplot as plt
import numpy as np

# Define the exercise program
exercise_program = {
    'Day 1 - Cardio and Lower Body': [
        ('Jogging or brisk walking', 30),
        ('Squats', 4 * 12),
        ('Lunges', 4 * 12),
        ('Step-up', 3 * 15)
    ],
    'Day 2 - Upper Body': [
        ('Dumbbell Press', 4 * 10),
        ('Bent-over Rows', 4 * 12),
        ('Shoulder press', 3 * 10),
        ('Push-ups', 'max')
    ],
    'Day 3 - Cardio and Light Weights': [
        ('Stationary bike', 30),
        ('Light Weight Circuit', 3),
        ('Plank', 3)
    ],
    'Day 4 - Core': [
        ('Leg raises', 4 * 15),
        ('Russian twists', 3 * 30),
        ('Climbers', 3),
        ('Cycling', 4 * 20)
    ],
    'Day 5 - Rest or Light Cardio': [
        ('Light jogging or brisk walking', 30)
    ]
}

# Create a bar chart for the exercise program
fig, ax = plt.subplots(figsize=(10, 6))

# Colors for each day
colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFD700']

# Plot each day's exercises
for i, (day, exercises) in enumerate(exercise_program.items()):
    y_pos = np.arange(len(exercises))
    reps = [e[1] if isinstance(e[1], int) else 0 for e in exercises]  # Use 0 for 'max' to plot
    ax.barh(y_pos + i * 0.2, reps, height=0.2, color=colors[i], label=day)

# Set labels and title
ax.set_yticks(np.arange(len(exercises)) + 0.3)
ax.set_yticklabels([e[0] for e in exercises])
ax.set_xlabel('Repetitions/Minutes')
ax.set_title('Weekly Exercise Program')
ax.legend()

plt.tight_layout()
plt.show()