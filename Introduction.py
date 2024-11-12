import streamlit as st
# Main Heading
st.write("# Autism Spectrum Disorder Classification")

# Introduction
st.write("### Introduction")
st.write(
    """
    Autism spectrum disorder (ASD) is a neurodevelopmental condition that profoundly impacts communication, interpersonal relationships, and behavior. 
    Recent research is focused on identifying **atypical speech patterns** associated with ASD to improve diagnostic accuracy and treatment approaches.
    """
)

# Language Patterns in ASD
st.write("### Language Patterns in ASD")
st.write(
    """
    Individuals with ASD often exhibit unique language characteristics, such as differences in:
    - **Prosody**: the rhythm and melody of speech
    - **Pace**: how fast or slow they speak
    - **Accent**: variations in speech sounds
    - **Language usage**: choice of words and sentence structure

    These distinct patterns make it challenging to classify speech in individuals with ASD, as each person's communication style may vary widely.
    """
)

st.write("### Purpose of Classification")
st.write(
    """
    By analyzing responses to these questions, researchers aim to improve:
    - **ASD diagnosis accuracy**
    - **Personalized support** for ASD individuals
    - **Understanding of ASD-related communication patterns**, enhancing early interventions and long-term support.
    """
)

# Dataset Overview
st.write("### Dataset Overview")
st.write(
    """
    This study uses data from the **UCI Machine Learning Repository**. The dataset includes information from **three groups** (children, adolescents, and adults) and 
    consists of **21 columns** that capture responses to ASD screening questions, along with demographic information and diagnostic results.
    """
)

# Dataset Structure
st.write("#### Dataset Structure")
st.write(
    """
    Each dataset contains the following columns:

    - **A1-A10**: Screening question scores that evaluate specific behaviors and communication traits related to ASD.
    
    - **Age**: The age of the individual, indicating their developmental stage.
    
    - **Gender**: The gender of the individual, noted as 'Male' or 'Female'.
    
    - **Ethnicity**: The ethnic background of the individual, which may help assess any cultural influence on behavior.
    
    - **Jaundice**: Indicates whether the individual had jaundice at birth ('Yes' or 'No'), as some studies suggest a potential link to ASD.
    
    - **Autism**: Family history of autism, recorded as 'Yes' or 'No'.
    
    - **Country of Residence**: The individual’s country of residence, which may provide insights into cultural and environmental factors.
    
    - **Used App Before**: Indicates whether the individual has used the ASD screening app before ('Yes' or 'No').
    
    - **Result**: The screening result score, summarizing responses to the questions.
    
    - **Age Description**: General descriptor of the age group (e.g., ‘Child’, ‘Adolescent’, or ‘Adult’).
    
    - **Relation**: The respondent’s relationship to the individual, which may be relevant for interpreting responses in child and adolescent groups.
    
    - **Class/ASD**: The target variable, representing whether the individual is classified as having ASD (binary: 1 for ASD, 0 for no ASD).
    """
)

# Screening Questions Breakdown
st.write("### Screening Questions (A1 - A10)")
st.write("Below are the 10 screening questions used to evaluate behavioral patterns, with variations tailored to each age group.")

# Question Breakdown
questions = [
    ("Q1", 
     "When you call your child's name, does they look at you? (Toddler) | "
     "Can they detect minor sounds others don't? (Child, Adolescent) | "
     "Are they always looking for patterns in things? (Adult)"),
    
    ("Q2", 
     "How quickly can they make eye contact with you? (Toddler) | "
     "Do they focus more on the big picture than the details? (Child, Adolescent, Adult)"),
    
    ("Q3", 
     "When your child wants something (like a toy), do they point to it? (Toddler) | "
     "Can they focus on others' conversations in social settings? (Child, Adolescent) | "
     "Is multitasking easy for them? (Adult)"),
    
    ("Q4", 
     "Is your child interested in sharing activities with you? (Toddler) | "
     "Can they quickly switch between different activities? (Child, Adolescent) | "
     "Can they resume an interrupted activity quickly? (Adult)"),
    
    ("Q5", 
     "Does your child pretend to care for their toys? (Toddler) | "
     "Do they find it hard to keep up conversations with friends? (Child, Adolescent) | "
     "Is it easy to read between the lines in conversations? (Adult)"),
    
    ("Q6", 
     "Is your child interested in searching for lost items with you? (Toddler) | "
     "Do they engage enthusiastically in social conversations? (Child, Adolescent) | "
     "Can they sense when someone is unwilling to listen? (Adult)"),
    
    ("Q7", 
     "Can your child console a family member if upset? (Toddler) | "
     "Do they assign goals to characters when reading stories? (Child) | "
     "Did they enjoy playing with other kids at a young age? (Adolescent) | "
     "Do they struggle to understand motives of characters in stories? (Adult)"),
    
    ("Q8", 
     "What were your child's first words? (Toddler) | "
     "Did they play fantasy games with friends? (Child) | "
     "How well can they imagine being someone else? (Adolescent) | "
     "Do they feel passionate about learning different topics (plants, cars, animals)? (Adult)"),
    
    ("Q9", 
     "Does your child use simple cues like goodbye? (Toddler) | "
     "Can they interpret facial expressions? (Child) | "
     "Are social events easy to read? (Adolescent) | "
     "Can they understand others' thoughts from facial expressions? (Adult)"),
    
    ("Q10", 
     "Does your child look at unnecessary details? (Toddler) | "
     "Do they struggle to make new friends? (Child, Adolescent) | "
     "Do they have difficulties understanding others' beliefs? (Adult)")
]

# Display questions
for q_num, q_text in questions:
    st.write(f"**{q_num}:** {q_text}")
