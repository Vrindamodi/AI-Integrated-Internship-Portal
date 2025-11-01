# Sample data for testing the Smart Allocation Engine

# Sample Candidates Data
candidates_sample = [
    {
        "id": "C001",
        "name": "Rajesh Kumar",
        "qualification": "btech",
        "skills": "python;machine learning;data analysis;sql",
        "profile_text": "Computer Science graduate with strong programming skills and machine learning experience",
        "district": "bangalore",
        "category": "gen",
        "past_participation": False,
        "cgpa": 8.5,
        "distance": 15.5,
        "age": 23,
        "income": 450000,
        "gender": "male",
        "pwd": False
    },
    {
        "id": "C002",
        "name": "Priya Sharma",
        "qualification": "bsc",
        "skills": "web development;javascript;react;html;css",
        "profile_text": "Web developer with expertise in modern frontend technologies",
        "district": "mumbai",
        "category": "sc",
        "past_participation": False,
        "cgpa": 7.8,
        "distance": 25.0,
        "age": 22,
        "income": 350000,
        "gender": "female",
        "pwd": False
    },
    {
        "id": "C003",
        "name": "Amit Singh",
        "qualification": "bca",
        "skills": "java;spring boot;mysql;microservices",
        "profile_text": "Backend developer specializing in Java and Spring framework",
        "district": "delhi",
        "category": "obc",
        "past_participation": True,
        "cgpa": 8.2,
        "distance": 10.0,
        "age": 24,
        "income": 500000,
        "gender": "male",
        "pwd": False
    },
    {
        "id": "C004",
        "name": "Sunita Devi",
        "qualification": "btech",
        "skills": "android;kotlin;firebase;ui/ux",
        "profile_text": "Mobile app developer with Android and UI/UX design skills",
        "district": "rural_pune",
        "category": "st",
        "past_participation": False,
        "cgpa": 7.5,
        "distance": 45.0,
        "age": 21,
        "income": 250000,
        "gender": "female",
        "pwd": False
    },
    {
        "id": "C005",
        "name": "Vikram Patel",
        "qualification": "btech",
        "skills": "cloud computing;aws;docker;kubernetes",
        "profile_text": "DevOps engineer with cloud computing and containerization expertise",
        "district": "hyderabad",
        "category": "gen",
        "past_participation": False,
        "cgpa": 9.1,
        "distance": 20.0,
        "age": 25,
        "income": 600000,
        "gender": "male",
        "pwd": False
    }
]

# Sample Internships Data
internships_sample = [
    {
        "id": "I001",
        "org": "TechCorp Solutions",
        "role": "Software Development Intern",
        "required_skills": "python;java;javascript;sql",
        "description": "Full-stack development internship focusing on web applications and database design",
        "min_qualification": "btech",
        "capacity": 3,
        "district": "bangalore",
        "sector": "technology",
        "reserved_percent": 30
    },
    {
        "id": "I002",
        "org": "DataInsights Ltd",
        "role": "Data Science Intern",
        "required_skills": "python;machine learning;statistics;data analysis",
        "description": "Data science internship involving machine learning model development and data visualization",
        "min_qualification": "bsc",
        "capacity": 2,
        "district": "mumbai",
        "sector": "data_science",
        "reserved_percent": 25
    },
    {
        "id": "I003",
        "org": "MobileFirst Inc",
        "role": "Mobile App Development Intern",
        "required_skills": "android;kotlin;java;firebase",
        "description": "Mobile application development internship with focus on Android platform",
        "min_qualification": "bca",
        "capacity": 2,
        "district": "delhi",
        "sector": "mobile_development",
        "reserved_percent": 20
    },
    {
        "id": "I004",
        "org": "CloudTech Systems",
        "role": "DevOps Intern",
        "required_skills": "aws;docker;kubernetes;linux",
        "description": "DevOps internship focusing on cloud infrastructure and containerization",
        "min_qualification": "btech",
        "capacity": 1,
        "district": "hyderabad",
        "sector": "cloud_computing",
        "reserved_percent": 0
    },
    {
        "id": "I005",
        "org": "WebCraft Studios",
        "role": "Frontend Development Intern",
        "required_skills": "react;javascript;html;css;ui/ux",
        "description": "Frontend development internship with modern web technologies and UI/UX design",
        "min_qualification": "bsc",
        "capacity": 2,
        "district": "pune",
        "sector": "web_development",
        "reserved_percent": 40
    }
]

# Sample Matching Request
matching_request_sample = {
    "aspirational_districts": ["rural_pune", "rural_mumbai"],
    "beneficiary_categories": ["sc", "st", "obc"],
    "weights": {
        "semantic": 0.55,
        "qualification": 0.15,
        "cgpa": 0.15,
        "location": 0.08,
        "past_penalty": 0.05,
        "aff_boost": 0.02
    },
    "age_min": 21,
    "age_max": 30,
    "use_embeddings": True
}
