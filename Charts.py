import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.title("Exploratory Data Analysis")
#Dataset Processing

def encode_handle_unknown(train):
    for col in train:
        le = LabelEncoder()
        le.fit(train[col])

        train[col] = le.transform(train[col])

    return train    

df = pd.read_csv('projectData.csv')
df = df.replace('?', np.nan)

df['age'] = pd.to_numeric(df['age'], errors = 'coerce')
df['age'] = df['age'].fillna(0).astype('int64')
df['age'] = df['age'].replace(0, np.nan)

df['age'] = df['age'].fillna(df['age'].mean())
df['ethnicity'] = df['ethnicity'].fillna(df['ethnicity'].mode()[0])
df['relation'] = df['relation'].fillna(df['relation'].mode()[0])

df = encode_handle_unknown(df)

st.header("Heatmap")

correlation_matrix = df.corr()
plt.figure(figsize=(30, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
st.pyplot(plt)

st.divider()

st.header("Distribution of Ages")
df = df[df['age'] != 343]
plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Ages')
st.pyplot(plt)

st.divider()

st.header("Gender Distribution")
df['gender'] = df['gender'].replace({'f': 'Female', 'm': 'Male'})
gender_counts = df['gender'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightpink'])
plt.title('Gender Distribution')
st.pyplot(plt)


st.divider()

st.header("Distribution of A6 and A9 Scores by Result")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.violinplot(data=df, x='result', y='A6_Score', palette='muted')
plt.title('Distribution of A6 Score by Result')
plt.xlabel('Result')
plt.ylabel('A6 Score')
plt.subplot(1, 2, 2)
sns.violinplot(data=df, x='result', y='A9_Score', palette='muted')
plt.title('Distribution of A9 Score by Result')
plt.xlabel('Result')
plt.ylabel('A9 Score')
plt.tight_layout()
st.pyplot(plt)

st.divider()


st.header("Count of Class/ASD for Each Result (0 and 1)")
count_df = df.groupby(['result', 'Class/ASD']).size().unstack(fill_value=0)
count_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['lightblue', 'lightcoral'])
plt.xlabel('Result')
plt.ylabel('Count of Class/ASD')
plt.title('Count of Class/ASD for Each Result (0 and 1)')
plt.xticks(rotation=0)
plt.legend(title='Class/ASD')
plt.tight_layout()
st.pyplot(plt)

st.divider()


st.header("Clustered Bar Chart of Age Groups vs Class/ASD ")
df_filtered = df[df['age'] != 383]
bins = [0, 10, 20, 30, 40, 50, 60, 70]
labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70']
df_filtered['Age Group'] = pd.cut(df_filtered['age'], bins=bins, labels=labels, right=False)
age_class_counts = df_filtered.groupby(['Age Group', 'Class/ASD']).size().unstack(fill_value=0)
age_class_counts.plot(kind='bar', stacked=False, figsize=(12, 6), colormap='viridis')
plt.title('Clustered Bar Chart of Age Groups vs Class/ASD (Excluding Age 383)')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Class/ASD', labels=['0', '1'])
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)


st.divider()

st.header("Pie Charts for A1 to A9 Scores")
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()
for i, score in enumerate([f'A{j}_Score' for j in range(1, 10)]):
    counts = df[score].value_counts()
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    axes[i].axis('equal')
    axes[i].set_title(f'Pie Chart for {score}')
plt.tight_layout()
st.pyplot(plt)

st.divider()


st.header("Distribution of Ethnicity with Class/ASD")
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='ethnicity', hue='Class/ASD', palette='viridis')
plt.xlabel('Ethnicity')
plt.ylabel('Count')
plt.title('Distribution of Ethnicity with Class/ASD')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(plt)

st.divider()

st.header("Age Distribution by Gender")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='gender', y='age', palette='coolwarm')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.title('Age Distribution by Gender')
st.pyplot(plt)

st.divider()


st.header("Distribution of Class/ASD by Top 10 Countries of Residence")
country_class_counts = df.groupby(['contry_of_res', 'Class/ASD']).size().unstack(fill_value=0)
top_10_countries = country_class_counts.sum(axis=1).nlargest(10).index
top_10_country_class_counts = country_class_counts.loc[top_10_countries]
top_10_country_class_counts.plot(kind='bar', stacked=False, figsize=(14, 7), color=['skyblue', 'salmon'])
plt.xlabel('Country of Residence')
plt.ylabel('Count')
plt.title('Distribution of Class/ASD by Top 10 Countries of Residence')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Class/ASD', labels=['0', '1'])
plt.tight_layout()
st.pyplot(plt)

st.divider()

st.header("Distribution of Class/ASD for Males and Females")
male_df = df[df['gender'] == 1]
female_df = df[df['gender'] == 0]
male_counts = male_df['Class/ASD'].value_counts()
female_counts = female_df['Class/ASD'].value_counts()
print(df['gender'].unique())
fig, axes = plt.subplots(1, 2, figsize=(15, 15))
axes[0].pie(male_counts, labels=male_counts.index, autopct='%1.1f%%', colors=['lightblue', 'salmon'])
axes[0].set_title('Distribution of Class/ASD for Males')
axes[1].pie(female_counts, labels=female_counts.index, autopct='%1.1f%%', colors=['lightpink', 'salmon'])
axes[1].set_title('Distribution of Class/ASD for Females')
plt.tight_layout()
st.pyplot(plt)
