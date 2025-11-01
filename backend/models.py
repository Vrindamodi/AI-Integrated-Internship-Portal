from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
import enum

Base = declarative_base()

# User Role Enum
class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    applications = relationship("Application", back_populates="user")
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    placements = relationship("Placement", back_populates="user")
class Candidate(Base):
    __tablename__ = "candidates"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    qualification = Column(String)
    skills = Column(Text)  # semicolon-separated skills
    profile_text = Column(Text)
    district = Column(String)
    category = Column(String, default="gen")
    past_participation = Column(Boolean, default=False)
    cgpa = Column(Float)
    distance = Column(Float)
    age = Column(Integer)
    income = Column(Float)
    gender = Column(String)
    pwd = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Internship(Base):
    __tablename__ = "internships"
    
    id = Column(String, primary_key=True)
    org = Column(String, nullable=False)
    role = Column(String, nullable=False)
    required_skills = Column(Text)  # semicolon-separated skills
    description = Column(Text)
    min_qualification = Column(String)
    capacity = Column(Integer, default=1)
    district = Column(String)
    sector = Column(String)
    reserved_percent = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    placements = relationship("Placement", back_populates="internship")
    applications = relationship("Application", back_populates="internship")

class Placement(Base):
    __tablename__ = "placements"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    internship_id = Column(String, ForeignKey("internships.id"), nullable=False)
    total_score = Column(Float)
    semantic_score = Column(Float)
    qualification_score = Column(Float)
    cgpa_score = Column(Float)
    location_score = Column(Float)
    past_penalty = Column(Float)
    aff_boost = Column(Float)
    status = Column(String, default="assigned")  # assigned, waitlist, opted_out
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="placements")
    internship = relationship("Internship", back_populates="placements")

class Application(Base):
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    internship_id = Column(String, ForeignKey("internships.id"), nullable=False)
    status = Column(String, default="pending")  # pending, accepted, rejected, withdrawn
    applied_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="applications")
    internship = relationship("Internship", back_populates="applications")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    name = Column(String, nullable=False)
    qualification = Column(String)
    skills = Column(Text)  # semicolon-separated skills
    profile_text = Column(Text)
    district = Column(String)
    category = Column(String, default="gen")
    past_participation = Column(Boolean, default=False)
    cgpa = Column(Float)
    age = Column(Integer)
    income = Column(Float)
    gender = Column(String)
    pwd = Column(Boolean, default=False)
    # New fields from registration form
    phone_number = Column(String)
    city = Column(String)
    university_name = Column(String)
    degree = Column(String)
    major = Column(String)
    graduation_year = Column(String)
    resume_file_path = Column(String)  # Path to uploaded resume file
    portfolio_link = Column(String)
    certifications = Column(Text)  # semicolon-separated certifications
    job_type_preference = Column(String, default="on-site")  # on-site, remote, hybrid
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="profile")

# Pydantic Models for API
class CandidateBase(BaseModel):
    name: str
    qualification: Optional[str] = None
    skills: Optional[str] = None
    profile_text: Optional[str] = None
    district: Optional[str] = None
    category: str = "gen"
    past_participation: bool = False
    cgpa: Optional[float] = None
    distance: Optional[float] = None
    age: Optional[int] = None
    income: Optional[float] = None
    gender: Optional[str] = None
    pwd: bool = False

class CandidateCreate(CandidateBase):
    id: str

class CandidateUpdate(BaseModel):
    name: Optional[str] = None
    qualification: Optional[str] = None
    skills: Optional[str] = None
    profile_text: Optional[str] = None
    district: Optional[str] = None
    category: Optional[str] = None
    past_participation: Optional[bool] = None
    cgpa: Optional[float] = None
    distance: Optional[float] = None
    age: Optional[int] = None
    income: Optional[float] = None
    gender: Optional[str] = None
    pwd: Optional[bool] = None

class CandidateResponse(CandidateBase):
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class InternshipBase(BaseModel):
    org: str
    role: str
    required_skills: Optional[str] = None
    description: Optional[str] = None
    min_qualification: Optional[str] = None
    capacity: int = 1
    district: Optional[str] = None
    sector: Optional[str] = None
    reserved_percent: int = 0

class InternshipCreate(InternshipBase):
    id: str

class InternshipUpdate(BaseModel):
    org: Optional[str] = None
    role: Optional[str] = None
    required_skills: Optional[str] = None
    description: Optional[str] = None
    min_qualification: Optional[str] = None
    capacity: Optional[int] = None
    district: Optional[str] = None
    sector: Optional[str] = None
    reserved_percent: Optional[int] = None

class InternshipResponse(InternshipBase):
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class PlacementResponse(BaseModel):
    id: int
    user_id: int
    internship_id: str
    total_score: Optional[float] = None
    semantic_score: Optional[float] = None
    qualification_score: Optional[float] = None
    cgpa_score: Optional[float] = None
    location_score: Optional[float] = None
    past_penalty: Optional[float] = None
    aff_boost: Optional[float] = None
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class MatchingRequest(BaseModel):
    aspirational_districts: Optional[List[str]] = []
    beneficiary_categories: List[str] = ["sc", "st"]
    weights: Optional[dict] = None
    age_min: Optional[int] = 21
    age_max: Optional[int] = 30
    use_embeddings: bool = True

class MatchingResponse(BaseModel):
    success: bool
    message: str
    total_candidates: int
    total_internships: int
    matched_count: int
    waitlist_count: int
    placements: List[PlacementResponse]

class WaitlistPromotionRequest(BaseModel):
    internship_id: str
    opted_out_user_id: Optional[int] = None

class WaitlistPromotionResponse(BaseModel):
    success: bool
    message: str
    promoted_candidate_id: Optional[str] = None

class IndividualMatchingRequest(BaseModel):
    aspirational_districts: Optional[List[str]] = []
    beneficiary_categories: List[str] = ["sc", "st"]
    weights: Optional[dict] = None
    age_min: Optional[int] = 21
    age_max: Optional[int] = 30
    use_embeddings: bool = True
    max_candidates: Optional[int] = 10  # Limit number of top candidates returned

class CandidateScore(BaseModel):
    user_id: int
    total_score: float
    semantic_score: float
    qualification_score: float
    cgpa_score: float
    location_score: float
    past_penalty: float
    aff_boost: float
    user_profile: Optional[dict] = None
    user: Optional[dict] = None

class IndividualMatchingResponse(BaseModel):
    success: bool
    message: str
    internship_id: str
    total_candidates_evaluated: int
    eligible_candidates: int
    top_candidates: List[CandidateScore]
    internship_details: Optional[dict] = None

# Authentication Models
class UserBase(BaseModel):
    username: str
    email: str
    role: UserRole = UserRole.USER

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

# Application Models
class ApplicationBase(BaseModel):
    internship_id: str

class ApplicationCreate(ApplicationBase):
    pass

class ApplicationUpdate(BaseModel):
    status: Optional[str] = None

class ApplicationResponse(ApplicationBase):
    id: int
    user_id: int
    status: str
    applied_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# User Profile Models
class UserProfileBase(BaseModel):
    name: str
    qualification: Optional[str] = None
    skills: Optional[str] = None
    profile_text: Optional[str] = None
    district: Optional[str] = None
    category: str = "gen"
    past_participation: bool = False
    cgpa: Optional[float] = None
    age: Optional[int] = None
    income: Optional[float] = None
    gender: Optional[str] = None
    pwd: bool = False
    # New fields from registration form
    phone_number: Optional[str] = None
    city: Optional[str] = None
    university_name: Optional[str] = None
    degree: Optional[str] = None
    major: Optional[str] = None
    graduation_year: Optional[str] = None
    resume_file_path: Optional[str] = None
    portfolio_link: Optional[str] = None
    certifications: Optional[str] = None
    job_type_preference: str = "on-site"

class UserProfileCreate(UserProfileBase):
    pass

class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    qualification: Optional[str] = None
    skills: Optional[str] = None
    profile_text: Optional[str] = None
    district: Optional[str] = None
    category: Optional[str] = None
    past_participation: Optional[bool] = None
    cgpa: Optional[float] = None
    age: Optional[int] = None
    income: Optional[float] = None
    gender: Optional[str] = None
    pwd: Optional[bool] = None
    # New fields from registration form
    phone_number: Optional[str] = None
    city: Optional[str] = None
    university_name: Optional[str] = None
    degree: Optional[str] = None
    major: Optional[str] = None
    graduation_year: Optional[str] = None
    resume_file_path: Optional[str] = None
    portfolio_link: Optional[str] = None
    certifications: Optional[str] = None
    job_type_preference: Optional[str] = None

class UserProfileResponse(UserProfileBase):
    user_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
