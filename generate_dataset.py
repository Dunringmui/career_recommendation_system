import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define categories
streams = ['Science', 'Commerce', 'Arts']
activities = ['Code', 'Design', 'Debate', 'Care', 'Media', 'Organize', 'Research', 'Outdoor']
interests = ['Programming', 'Reading', 'Drawing', 'Public Speaking', 'Helping', 'Writing', 'Designing', 'Exploring']
skills = ['Python', 'Creativity', 'Communication', 'Empathy', 'Leadership', 'Observation', 'Problem Solving', 'Teamwork']
subjects = ['Physics', 'Chemistry', 'Biology', 'Mathematics', 'Computer Science', 'Economics', 
            'Business Studies', 'History', 'Political Science', 'Psychology']
envs = ['Solo', 'Team', 'Outdoor', 'Structured']
styles = ['Visual', 'Reading', 'Hands-On', 'Listening']

# Define realistic stream-activity-career mappings
careers_map = {
    ('Science', 'Code'): 'Software Developer',
    ('Science', 'Research'): 'Research Scientist',
    ('Science', 'Care'): 'Doctor',
    ('Science', 'Design'): 'Electrical Engineer',
    ('Science', 'Organize'): 'Pharmacist',
    ('Science', 'Media'): 'Science Journalist',
    ('Science', 'Outdoor'): 'Civil Engineer',
    ('Commerce', 'Organize'): 'Management',
    ('Commerce', 'Care'): 'Accountant',
    ('Commerce', 'Media'): 'Marketing Specialist',
    ('Commerce', 'Design'): 'Chef',
    ('Commerce', 'Code'): 'Financial Analyst',
    ('Arts', 'Debate'): 'Lawyer',
    ('Arts', 'Media'): 'Mass Communicator',
    ('Arts', 'Design'): 'Interior Designer',
    ('Arts', 'Care'): 'Psychologist',
    ('Arts', 'Outdoor'): 'Tourism Guide',
    ('Arts', 'Research'): 'Linguist',
    ('Arts', 'Organize'): 'Social Worker'
}

# Generate synthetic data
rows = []
for (stream, activity), career in careers_map.items():
    for _ in range(55):  # ~1000 total rows
        rows.append({
            "Stream": stream,
            "Activity": activity,
            "Interest1": random.choice(interests),
            "Interest2": random.choice(interests),
            "Skill1": random.choice(skills),
            "Skill2": random.choice(skills),
            "Subject1": random.choice(subjects),
            "Subject2": random.choice(subjects),
            "Numerical": round(np.clip(np.random.normal(75, 10), 40, 100), 2),
            "Logical": int(np.clip(np.random.normal(7, 2), 1, 10)),
            "English": int(np.clip(np.random.normal(7, 2), 1, 10)),
            "Personality": int(np.clip(np.random.normal(35, 5), 20, 50)),
            "PreferredEnv": random.choice(envs),
            "StudyStyle": random.choice(styles),
            "Career": career
        })

# Create DataFrame and save
df = pd.DataFrame(rows)
df.to_csv("career_dataset.csv", index=False)
print("✅ Dataset created: career_dataset.csv")
