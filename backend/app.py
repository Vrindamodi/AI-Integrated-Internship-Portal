from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import os
import uuid
from datetime import timedelta

from models import (
    Candidate, Internship, Placement, User, UserProfile, Application,
    CandidateCreate, CandidateUpdate, CandidateResponse,
    InternshipCreate, InternshipUpdate, InternshipResponse,
    PlacementResponse, MatchingRequest, MatchingResponse,
    WaitlistPromotionRequest, WaitlistPromotionResponse,
    IndividualMatchingRequest, IndividualMatchingResponse, CandidateScore,
    UserCreate, UserResponse, UserUpdate, LoginRequest, Token, UserRole,
    UserProfileCreate, UserProfileUpdate, UserProfileResponse,
    ApplicationCreate, ApplicationUpdate, ApplicationResponse
)
from database import get_db, create_tables
from allocation_engine import (
    CandidateData, InternshipData, EmbedModel,
    compute_scores_components, two_phase_matching, build_waitlist_from_scores,
    is_eligible, DEFAULT_WEIGHTS, HAS_ST, create_candidates_from_applications
)
from auth import (
    authenticate_user, create_user, get_current_active_user, 
    get_current_admin_user, get_current_user_or_admin,
    create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Smart Allocation Engine API",
    description="AI-powered internship allocation system with fairness considerations",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "User authentication and management endpoints",
        },
        {
            "name": "User Profiles",
            "description": "User profile management endpoints",
        },
        {
            "name": "Applications",
            "description": "Internship application management endpoints",
        },
        {
            "name": "Candidates",
            "description": "Candidate management endpoints (admin only)",
        },
        {
            "name": "Internships",
            "description": "Internship management endpoints",
        },
        {
            "name": "Matching",
            "description": "Smart allocation algorithm endpoints",
        },
        {
            "name": "Placements",
            "description": "Placement management endpoints",
        },
        {
            "name": "Waitlist",
            "description": "Waitlist management endpoints",
        },
        {
            "name": "Health",
            "description": "Health check endpoints",
        },
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files for serving uploaded files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Security scheme for Swagger UI
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Smart Allocation Engine API",
        version="1.0.0",
        description="AI-powered internship allocation system with fairness considerations",
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your JWT token. You can get this token by calling the /auth/login endpoint."
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    # Remove security from login and register endpoints
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method in ["get", "post", "put", "delete"]:
                endpoint = openapi_schema["paths"][path][method]
                # Remove security requirement from login and register endpoints
                if path.endswith("/auth/login") or path.endswith("/auth/register"):
                    endpoint["security"] = []
                # Add security requirement to all other endpoints
                elif ("tags" in endpoint and 
                      any(tag in ["Candidates", "Internships", "Matching", "Placements", "Waitlist", "Authentication", "User Profiles", "Applications"] for tag in endpoint["tags"])):
                    endpoint["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Global embedding model
embedding_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize database and embedding model on startup"""
    create_tables()
    logger.info("Database tables created")
    
    # Initialize embedding model
    global embedding_model
    if HAS_ST:
        try:
            embedding_model = EmbedModel()
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            embedding_model = None
    else:
        logger.info("Sentence transformers not available, using skill overlap fallback")

# -----------------------
# Authentication Endpoints
# -----------------------
@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user (admin only)"""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username or email already registered"
        )
    
    # Create new user
    db_user = create_user(
        db=db,
        username=user.username,
        email=user.email,
        password=user.password,
        role=user.role
    )
    return db_user

@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login_user(login_data: LoginRequest, db: Session = Depends(get_db)):
    """Login user and return access token"""
    user = authenticate_user(db, login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user

@app.put("/auth/users/{user_id}", response_model=UserResponse, tags=["Authentication"])
async def update_user(
    user_id: int, 
    user_update: UserUpdate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Update user information (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    update_data = user_update.dict(exclude_unset=True)
    
    # Hash password if provided
    if "password" in update_data:
        from auth import get_password_hash
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    for field, value in update_data.items():
        setattr(user, field, value)
    
    db.commit()
    db.refresh(user)
    return user

@app.get("/auth/users", response_model=List[UserResponse], tags=["Authentication"])
async def get_users(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Get all users (admin only)"""
    users = db.query(User).offset(skip).limit(limit).all()
    return users

# -----------------------
# User Profile Endpoints
# -----------------------
@app.post("/profile/", response_model=UserProfileResponse, tags=["User Profiles"])
async def create_user_profile(
    profile: UserProfileCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create or update user profile"""
    # Check if profile already exists
    existing_profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if existing_profile:
        # Update existing profile
        update_data = profile.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(existing_profile, field, value)
        db.commit()
        db.refresh(existing_profile)
        return existing_profile
    else:
        # Create new profile
        db_profile = UserProfile(user_id=current_user.id, **profile.dict())
        db.add(db_profile)
        db.commit()
        db.refresh(db_profile)
        return db_profile

@app.get("/profile/", response_model=UserProfileResponse, tags=["User Profiles"])
async def get_user_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's profile"""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found. Please create your profile first.")
    return profile

@app.put("/profile/", response_model=UserProfileResponse, tags=["User Profiles"])
async def update_user_profile(
    profile_update: UserProfileUpdate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update user profile"""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found. Please create your profile first.")
    
    update_data = profile_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(profile, field, value)
    
    db.commit()
    db.refresh(profile)
    return profile

@app.delete("/profile/", tags=["User Profiles"])
async def delete_user_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete user profile"""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    db.delete(profile)
    db.commit()
    return {"message": "Profile deleted successfully"}

@app.post("/profile/upload-resume", tags=["User Profiles"])
async def upload_resume(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Upload resume file for user profile"""
    # Validate file type
    allowed_types = ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only PDF, DOC, and DOCX files are allowed."
        )
    
    # Validate file size (max 10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )
    
    # Generate unique filename
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'pdf'
    unique_filename = f"{current_user.id}_{uuid.uuid4().hex}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Update user profile with file path
        profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
        if profile:
            # Delete old file if exists
            if profile.resume_file_path:
                old_file_path = os.path.join(UPLOAD_DIR, os.path.basename(profile.resume_file_path))
                if os.path.exists(old_file_path):
                    os.remove(old_file_path)
            
            profile.resume_file_path = f"/uploads/{unique_filename}"
            db.commit()
            db.refresh(profile)
        else:
            raise HTTPException(status_code=404, detail="Profile not found. Please create your profile first.")
        
        return {
            "message": "Resume uploaded successfully",
            "file_path": f"/uploads/{unique_filename}",
            "filename": file.filename
        }
        
    except Exception as e:
        # Clean up file if database update fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

# -----------------------
# Application Endpoints
# -----------------------
@app.post("/applications/", response_model=ApplicationResponse, tags=["Applications"])
async def create_application(
    application: ApplicationCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Apply to an internship"""
    # Check if user has a profile
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=400, detail="Please create your profile before applying to internships")
    
    # Check if internship exists
    internship = db.query(Internship).filter(Internship.id == application.internship_id).first()
    if not internship:
        raise HTTPException(status_code=404, detail="Internship not found")
    
    # Check if user already applied to this internship
    existing_application = db.query(Application).filter(
        Application.user_id == current_user.id,
        Application.internship_id == application.internship_id
    ).first()
    
    if existing_application:
        raise HTTPException(status_code=400, detail="You have already applied to this internship")
    
    # Create application
    db_application = Application(
        user_id=current_user.id,
        internship_id=application.internship_id
    )
    db.add(db_application)
    db.commit()
    db.refresh(db_application)
    return db_application

@app.get("/applications/", response_model=List[ApplicationResponse], tags=["Applications"])
async def get_user_applications(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's applications"""
    applications = db.query(Application).filter(Application.user_id == current_user.id).all()
    return applications

@app.get("/applications/{application_id}", response_model=ApplicationResponse, tags=["Applications"])
async def get_application(
    application_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific application"""
    application = db.query(Application).filter(
        Application.id == application_id,
        Application.user_id == current_user.id
    ).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    return application

@app.put("/applications/{application_id}", response_model=ApplicationResponse, tags=["Applications"])
async def update_application(
    application_id: int, 
    application_update: ApplicationUpdate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update application status (withdraw application)"""
    application = db.query(Application).filter(
        Application.id == application_id,
        Application.user_id == current_user.id
    ).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Only allow users to withdraw their applications
    if application_update.status and application_update.status != "withdrawn":
        raise HTTPException(status_code=400, detail="You can only withdraw your applications")
    
    update_data = application_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(application, field, value)
    
    db.commit()
    db.refresh(application)
    return application

@app.delete("/applications/{application_id}", tags=["Applications"])
async def delete_application(
    application_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete application"""
    application = db.query(Application).filter(
        Application.id == application_id,
        Application.user_id == current_user.id
    ).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    db.delete(application)
    db.commit()
    return {"message": "Application deleted successfully"}

# -----------------------
# Admin Application Management Endpoints
# -----------------------
@app.get("/admin/applications/", response_model=List[ApplicationResponse], tags=["Applications"])
async def get_all_applications(
    skip: int = 0, 
    limit: int = 100, 
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Get all applications with pagination and filtering (admin only)"""
    query = db.query(Application)
    
    if status:
        query = query.filter(Application.status == status)
    
    applications = query.offset(skip).limit(limit).all()
    return applications

@app.get("/admin/applications/{application_id}", response_model=ApplicationResponse, tags=["Applications"])
async def get_application_admin(
    application_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Get a specific application (admin only)"""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    return application

@app.put("/admin/applications/{application_id}", response_model=ApplicationResponse, tags=["Applications"])
async def update_application_admin(
    application_id: int, 
    application_update: ApplicationUpdate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Update application status (admin only)"""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    update_data = application_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(application, field, value)
    
    db.commit()
    db.refresh(application)
    return application

@app.delete("/admin/applications/{application_id}", tags=["Applications"])
async def delete_application_admin(
    application_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Delete application (admin only)"""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    db.delete(application)
    db.commit()
    return {"message": "Application deleted successfully"}

@app.get("/admin/applications/{application_id}/details", tags=["Applications"])
async def get_application_details_admin(
    application_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Get detailed application information including user profile and internship details (admin only)"""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Get user profile
    user_profile = db.query(UserProfile).filter(UserProfile.user_id == application.user_id).first()
    
    # Get internship details
    internship = db.query(Internship).filter(Internship.id == application.internship_id).first()
    
    # Get user details
    user = db.query(User).filter(User.id == application.user_id).first()
    
    return {
        "application": application,
        "user_profile": user_profile,
        "internship": internship,
        "user": user
    }

# -----------------------
# Candidate Endpoints (Admin Only)
# -----------------------
@app.post("/candidates/", response_model=CandidateResponse, tags=["Candidates"])
async def create_candidate(
    candidate: CandidateCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Create a new candidate (admin only)"""
    db_candidate = Candidate(**candidate.dict())
    db.add(db_candidate)
    db.commit()
    db.refresh(db_candidate)
    return db_candidate

@app.get("/candidates/", response_model=List[CandidateResponse], tags=["Candidates"])
async def get_candidates(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_or_admin)
):
    """Get all candidates with pagination (authenticated users)"""
    candidates = db.query(Candidate).offset(skip).limit(limit).all()
    return candidates

@app.get("/candidates/{candidate_id}", response_model=CandidateResponse, tags=["Candidates"])
async def get_candidate(
    candidate_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_or_admin)
):
    """Get a specific candidate by ID (authenticated users)"""
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate

@app.put("/candidates/{candidate_id}", response_model=CandidateResponse, tags=["Candidates"])
async def update_candidate(
    candidate_id: str, 
    candidate_update: CandidateUpdate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Update a candidate (admin only)"""
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    update_data = candidate_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(candidate, field, value)
    
    db.commit()
    db.refresh(candidate)
    return candidate

@app.delete("/candidates/{candidate_id}", tags=["Candidates"])
async def delete_candidate(
    candidate_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Delete a candidate (admin only)"""
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    db.delete(candidate)
    db.commit()
    return {"message": "Candidate deleted successfully"}

# -----------------------
# Internship Endpoints
# -----------------------
@app.post("/internships/", response_model=InternshipResponse, tags=["Internships"])
async def create_internship(
    internship: InternshipCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Create a new internship (admin only)"""
    db_internship = Internship(**internship.dict())
    db.add(db_internship)
    db.commit()
    db.refresh(db_internship)
    return db_internship

@app.get("/internships/", response_model=List[InternshipResponse], tags=["Internships"])
async def get_internships(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_or_admin)
):
    """Get all internships with pagination (authenticated users)"""
    internships = db.query(Internship).offset(skip).limit(limit).all()
    return internships

@app.get("/internships/{internship_id}", response_model=InternshipResponse, tags=["Internships"])
async def get_internship(
    internship_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_or_admin)
):
    """Get a specific internship by ID (authenticated users)"""
    internship = db.query(Internship).filter(Internship.id == internship_id).first()
    if not internship:
        raise HTTPException(status_code=404, detail="Internship not found")
    return internship

@app.get("/internships/{internship_id}/applications", response_model=List[ApplicationResponse], tags=["Internships"])
async def get_internship_applications(
    internship_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_or_admin)
):
    """Get applications for a specific internship (authenticated users)"""
    applications = db.query(Application).filter(Application.internship_id == internship_id).all()
    return applications

@app.put("/internships/{internship_id}", response_model=InternshipResponse, tags=["Internships"])
async def update_internship(
    internship_id: str, 
    internship_update: InternshipUpdate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Update an internship (admin only)"""
    internship = db.query(Internship).filter(Internship.id == internship_id).first()
    if not internship:
        raise HTTPException(status_code=404, detail="Internship not found")
    
    update_data = internship_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(internship, field, value)
    
    db.commit()
    db.refresh(internship)
    return internship

@app.get("/admin/internships/{internship_id}/stats", tags=["Internships"])
async def get_internship_stats(
    internship_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Get internship statistics including application counts (admin only)"""
    internship = db.query(Internship).filter(Internship.id == internship_id).first()
    if not internship:
        raise HTTPException(status_code=404, detail="Internship not found")
    
    # Get application counts by status
    total_applications = db.query(Application).filter(Application.internship_id == internship_id).count()
    pending_applications = db.query(Application).filter(
        Application.internship_id == internship_id,
        Application.status == "pending"
    ).count()
    accepted_applications = db.query(Application).filter(
        Application.internship_id == internship_id,
        Application.status == "accepted"
    ).count()
    rejected_applications = db.query(Application).filter(
        Application.internship_id == internship_id,
        Application.status == "rejected"
    ).count()
    withdrawn_applications = db.query(Application).filter(
        Application.internship_id == internship_id,
        Application.status == "withdrawn"
    ).count()
    
    return {
        "internship_id": internship_id,
        "total_applications": total_applications,
        "pending_applications": pending_applications,
        "accepted_applications": accepted_applications,
        "rejected_applications": rejected_applications,
        "withdrawn_applications": withdrawn_applications,
        "capacity": internship.capacity,
        "utilization_percentage": round((accepted_applications / internship.capacity * 100) if internship.capacity > 0 else 0, 2)
    }

@app.get("/admin/internships/{internship_id}/details", tags=["Internships"])
async def get_internship_details(
    internship_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Get detailed internship information with applications and statistics (admin only)"""
    internship = db.query(Internship).filter(Internship.id == internship_id).first()
    if not internship:
        raise HTTPException(status_code=404, detail="Internship not found")
    
    # Get all applications for this internship
    applications = db.query(Application).filter(Application.internship_id == internship_id).all()
    
    # Get user profiles and user details for applications
    application_details = []
    for app in applications:
        user_profile = db.query(UserProfile).filter(UserProfile.user_id == app.user_id).first()
        user = db.query(User).filter(User.id == app.user_id).first()
        
        # Convert SQLAlchemy objects to dictionaries properly
        user_profile_dict = None
        if user_profile:
            user_profile_dict = {
                "id": user_profile.id,
                "user_id": user_profile.user_id,
                "name": user_profile.name,
                "qualification": user_profile.qualification,
                "skills": user_profile.skills,
                "profile_text": user_profile.profile_text,
                "district": user_profile.district,
                "category": user_profile.category,
                "past_participation": user_profile.past_participation,
                "cgpa": user_profile.cgpa,
                "age": user_profile.age,
                "income": user_profile.income,
                "gender": user_profile.gender,
                "pwd": user_profile.pwd,
                "phone_number": user_profile.phone_number,
                "city": user_profile.city,
                "university_name": user_profile.university_name,
                "degree": user_profile.degree,
                "major": user_profile.major,
                "graduation_year": user_profile.graduation_year,
                "resume_file_path": user_profile.resume_file_path,
                "portfolio_link": user_profile.portfolio_link,
                "certifications": user_profile.certifications,
                "job_type_preference": user_profile.job_type_preference,
                "created_at": user_profile.created_at.isoformat() if user_profile.created_at else None,
                "updated_at": user_profile.updated_at.isoformat() if user_profile.updated_at else None
            }
        
        user_dict = None
        if user:
            user_dict = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value if user.role else None,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "updated_at": user.updated_at.isoformat() if user.updated_at else None
            }
        
        application_details.append({
            "id": app.id,
            "user_id": app.user_id,
            "internship_id": app.internship_id,
            "status": app.status,
            "applied_at": app.applied_at.isoformat() if app.applied_at else None,
            "updated_at": app.updated_at.isoformat() if app.updated_at else None,
            "user_profile": user_profile_dict,
            "user": user_dict
        })
    
    # Get statistics
    total_applications = len(applications)
    pending_applications = len([app for app in applications if app.status == "pending"])
    accepted_applications = len([app for app in applications if app.status == "accepted"])
    rejected_applications = len([app for app in applications if app.status == "rejected"])
    withdrawn_applications = len([app for app in applications if app.status == "withdrawn"])
    
    stats = {
        "internship_id": internship_id,
        "total_applications": total_applications,
        "pending_applications": pending_applications,
        "accepted_applications": accepted_applications,
        "rejected_applications": rejected_applications,
        "withdrawn_applications": withdrawn_applications,
        "capacity": internship.capacity,
        "utilization_percentage": round((accepted_applications / internship.capacity * 100) if internship.capacity > 0 else 0, 2)
    }
    
    return {
        "internship": {
            "id": internship.id,
            "org": internship.org,
            "role": internship.role,
            "required_skills": internship.required_skills,
            "description": internship.description,
            "min_qualification": internship.min_qualification,
            "capacity": internship.capacity,
            "district": internship.district,
            "sector": internship.sector,
            "reserved_percent": internship.reserved_percent,
            "created_at": internship.created_at.isoformat() if internship.created_at else None,
            "updated_at": internship.updated_at.isoformat() if internship.updated_at else None
        },
        "applications": application_details,
        "stats": stats
    }

# -----------------------
# Matching Algorithm Endpoint
# -----------------------
@app.post("/matching/run", response_model=MatchingResponse, tags=["Matching"])
async def run_matching(
    request: MatchingRequest, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Run the smart allocation algorithm (admin only)"""
    try:
        # Get all applications and internships
        applications_db = db.query(Application).filter(Application.status == "pending").all()
        internships_db = db.query(Internship).all()
        
        if not applications_db:
            raise HTTPException(status_code=400, detail="No pending applications found")
        if not internships_db:
            raise HTTPException(status_code=400, detail="No internships found")
        
        # Get user profiles for applications
        user_ids = [app.user_id for app in applications_db]
        profiles_db = db.query(UserProfile).filter(UserProfile.user_id.in_(user_ids)).all()
        
        if not profiles_db:
            raise HTTPException(status_code=400, detail="No user profiles found for applications")
        
        # Convert to algorithm format
        candidates_data = create_candidates_from_applications(applications_db, profiles_db)
        internships_data = [InternshipData(i) for i in internships_db]
        
        # Filter eligible candidates
        eligible_candidates = []
        ineligible_candidates = []
        
        for c in candidates_data:
            if is_eligible(c, age_min=request.age_min, age_max=request.age_max):
                eligible_candidates.append(c)
            else:
                ineligible_candidates.append(c)
                logger.info(f"Ineligible candidate {c.id}: qualification='{c.qualification}', age={c.age}, age_min={request.age_min}, age_max={request.age_max}")
        
        logger.info(f"Eligible candidates: {len(eligible_candidates)}, Ineligible: {len(ineligible_candidates)}")
        
        if not eligible_candidates:
            raise HTTPException(status_code=400, detail="No eligible candidates found")
        
        # Use provided weights or defaults
        weights = request.weights or DEFAULT_WEIGHTS.copy()
        
        # Prepare aspirational districts
        aspirational = set([d.strip().lower() for d in request.aspirational_districts]) if request.aspirational_districts else set()
        
        # Use embedding model if available and requested
        model = embedding_model if request.use_embeddings else None
        
        # Compute scores and components
        scores_base, components = compute_scores_components(
            eligible_candidates, internships_data, model, weights, aspirational
        )
        
        # Build final scores dict
        scores_final = {c.id: {} for c in eligible_candidates}
        for cid in scores_base:
            for iid in scores_base[cid]:
                comps = components.get(cid, {}).get(iid, {})
                final_val = comps.get("final", scores_base[cid].get(iid, 0.0))
                scores_final[cid][iid] = round(float(final_val), 6)
        
        # Define beneficiary predicate
        beneficiary_pred = lambda c: (
            c.category in request.beneficiary_categories or 
            (aspirational and c.district in aspirational) or 
            (c.district and "rural" in c.district)
        )
        
        # Run matching algorithm
        matching = two_phase_matching(
            eligible_candidates, internships_data, scores_final, components, beneficiary_pred
        )
        
        # Build waitlist
        waitlist = build_waitlist_from_scores(
            scores_final, internships_data, eligible_candidates, components, matching
        )
        
        # Clear existing placements
        db.query(Placement).delete()
        
        # Save placements to database
        placements = []
        for cid, iid in matching.items():
            comps = components.get(cid, {}).get(iid, {})
            placement = Placement(
                user_id=int(cid),  # cid is now user_id as string
                internship_id=iid,
                total_score=scores_final.get(cid, {}).get(iid, 0.0),
                semantic_score=comps.get("semantic", 0.0),
                qualification_score=comps.get("qualification", 0.0),
                cgpa_score=comps.get("cgpa", 0.0),
                location_score=comps.get("location", 0.0),
                past_penalty=comps.get("past_penalty", 0.0),
                aff_boost=comps.get("aff_boost", 0.0),
                status="assigned"
            )
            db.add(placement)
            placements.append(placement)
        
        # Save waitlist entries
        waitlist_count = 0
        for iid, cid_list in waitlist.items():
            for cid in cid_list:
                comps = components.get(cid, {}).get(iid, {})
                placement = Placement(
                    user_id=int(cid),  # cid is now user_id as string
                    internship_id=iid,
                    total_score=scores_final.get(cid, {}).get(iid, 0.0),
                    semantic_score=comps.get("semantic", 0.0),
                    qualification_score=comps.get("qualification", 0.0),
                    cgpa_score=comps.get("cgpa", 0.0),
                    location_score=comps.get("location", 0.0),
                    past_penalty=comps.get("past_penalty", 0.0),
                    aff_boost=comps.get("aff_boost", 0.0),
                    status="waitlist"
                )
                db.add(placement)
                waitlist_count += 1
        
        db.commit()
        
        # Convert placements to response format
        placement_responses = []
        for placement in placements:
            placement_responses.append(PlacementResponse.from_orm(placement))
        
        return MatchingResponse(
            success=True,
            message=f"Successfully matched {len(matching)} applicants",
            total_candidates=len(applications_db),
            total_internships=len(internships_db),
            matched_count=len(matching),
            waitlist_count=waitlist_count,
            placements=placement_responses
        )
        
    except Exception as e:
        logger.error(f"Error in matching algorithm: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")

# -----------------------
# Individual Internship Matching Endpoint
# -----------------------
@app.post("/matching/internship/{internship_id}", response_model=IndividualMatchingResponse, tags=["Matching"])
async def run_individual_matching(
    internship_id: str,
    request: IndividualMatchingRequest, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Run smart allocation for a specific internship to find best candidates (admin only)"""
    try:
        # Verify the internship exists
        internship_db = db.query(Internship).filter(Internship.id == internship_id).first()
        if not internship_db:
            raise HTTPException(status_code=404, detail="Internship not found")
        
        # Get all applications for this specific internship
        applications_db = db.query(Application).filter(
            Application.internship_id == internship_id,
            Application.status == "pending"
        ).all()
        
        if not applications_db:
            raise HTTPException(status_code=400, detail=f"No pending applications found for internship {internship_id}")
        
        # Get user profiles for applications
        user_ids = [app.user_id for app in applications_db]
        profiles_db = db.query(UserProfile).filter(UserProfile.user_id.in_(user_ids)).all()
        
        if not profiles_db:
            raise HTTPException(status_code=400, detail="No user profiles found for applications")
        
        # Convert to algorithm format
        candidates_data = create_candidates_from_applications(applications_db, profiles_db)
        internship_data = InternshipData(internship_db)
        
        # Filter eligible candidates
        eligible_candidates = []
        ineligible_candidates = []
        
        for c in candidates_data:
            if is_eligible(c, age_min=request.age_min, age_max=request.age_max):
                eligible_candidates.append(c)
            else:
                ineligible_candidates.append(c)
                logger.info(f"Ineligible candidate {c.id}: qualification='{c.qualification}', age={c.age}, age_min={request.age_min}, age_max={request.age_max}")
        
        logger.info(f"Eligible candidates for {internship_id}: {len(eligible_candidates)}, Ineligible: {len(ineligible_candidates)}")
        
        if not eligible_candidates:
            raise HTTPException(status_code=400, detail="No eligible candidates found")
        
        # Use provided weights or defaults
        weights = request.weights or DEFAULT_WEIGHTS.copy()
        
        # Prepare aspirational districts
        aspirational = set([d.strip().lower() for d in request.aspirational_districts]) if request.aspirational_districts else set()
        
        # Use embedding model if available and requested
        model = embedding_model if request.use_embeddings else None
        
        # Compute scores and components for this specific internship
        scores_base, components = compute_scores_components(
            eligible_candidates, [internship_data], model, weights, aspirational
        )
        
        # Build candidate scores list
        candidate_scores = []
        for candidate in eligible_candidates:
            if candidate.id in scores_base and internship_id in scores_base[candidate.id]:
                comps = components.get(candidate.id, {}).get(internship_id, {})
                final_score = comps.get("final", scores_base[candidate.id].get(internship_id, 0.0))
                
                # Get user profile and user details
                user_profile = db.query(UserProfile).filter(UserProfile.user_id == candidate.id).first()
                user = db.query(User).filter(User.id == candidate.id).first()
                
                # Convert SQLAlchemy objects to dictionaries properly
                user_profile_dict = None
                if user_profile:
                    user_profile_dict = {
                        "id": user_profile.id,
                        "user_id": user_profile.user_id,
                        "name": user_profile.name,
                        "qualification": user_profile.qualification,
                        "skills": user_profile.skills,
                        "profile_text": user_profile.profile_text,
                        "district": user_profile.district,
                        "category": user_profile.category,
                        "past_participation": user_profile.past_participation,
                        "cgpa": user_profile.cgpa,
                        "age": user_profile.age,
                        "income": user_profile.income,
                        "gender": user_profile.gender,
                        "pwd": user_profile.pwd,
                        "phone_number": user_profile.phone_number,
                        "city": user_profile.city,
                        "university_name": user_profile.university_name,
                        "degree": user_profile.degree,
                        "major": user_profile.major,
                        "graduation_year": user_profile.graduation_year,
                        "resume_file_path": user_profile.resume_file_path,
                        "portfolio_link": user_profile.portfolio_link,
                        "certifications": user_profile.certifications,
                        "job_type_preference": user_profile.job_type_preference,
                        "created_at": user_profile.created_at.isoformat() if user_profile.created_at else None,
                        "updated_at": user_profile.updated_at.isoformat() if user_profile.updated_at else None
                    }
                
                user_dict = None
                if user:
                    user_dict = {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "role": user.role.value if user.role else None,
                        "is_active": user.is_active,
                        "created_at": user.created_at.isoformat() if user.created_at else None,
                        "updated_at": user.updated_at.isoformat() if user.updated_at else None
                    }
                
                candidate_score = CandidateScore(
                    user_id=candidate.id,
                    total_score=round(float(final_score), 6),
                    semantic_score=round(float(comps.get("semantic", 0.0)), 6),
                    qualification_score=round(float(comps.get("qualification", 0.0)), 6),
                    cgpa_score=round(float(comps.get("cgpa", 0.0)), 6),
                    location_score=round(float(comps.get("location", 0.0)), 6),
                    past_penalty=round(float(comps.get("past_penalty", 0.0)), 6),
                    aff_boost=round(float(comps.get("aff_boost", 0.0)), 6),
                    user_profile=user_profile_dict,
                    user=user_dict
                )
                candidate_scores.append(candidate_score)
        
        # Sort by total score (highest first) and limit results
        candidate_scores.sort(key=lambda x: x.total_score, reverse=True)
        max_candidates = request.max_candidates or 10
        top_candidates = candidate_scores[:max_candidates]
        
        # Prepare internship details
        internship_details = {
            "id": internship_db.id,
            "org": internship_db.org,
            "role": internship_db.role,
            "required_skills": internship_db.required_skills,
            "description": internship_db.description,
            "min_qualification": internship_db.min_qualification,
            "capacity": internship_db.capacity,
            "district": internship_db.district,
            "sector": internship_db.sector,
            "reserved_percent": internship_db.reserved_percent,
            "created_at": internship_db.created_at.isoformat() if internship_db.created_at else None,
            "updated_at": internship_db.updated_at.isoformat() if internship_db.updated_at else None
        }
        
        return IndividualMatchingResponse(
            success=True,
            message=f"Successfully evaluated {len(eligible_candidates)} candidates for internship {internship_id}",
            internship_id=internship_id,
            total_candidates_evaluated=len(applications_db),
            eligible_candidates=len(eligible_candidates),
            top_candidates=top_candidates,
            internship_details=internship_details
        )
        
    except Exception as e:
        logger.error(f"Error in individual matching algorithm: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Individual matching failed: {str(e)}")

# -----------------------
# Placement Endpoints
# -----------------------
@app.get("/placements/", response_model=List[PlacementResponse], tags=["Placements"])
async def get_placements(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_or_admin)
):
    """Get all placements with pagination (authenticated users)"""
    placements = db.query(Placement).offset(skip).limit(limit).all()
    return placements

@app.get("/placements/user/{user_id}", response_model=List[PlacementResponse], tags=["Placements"])
async def get_user_placements(
    user_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_or_admin)
):
    """Get placements for a specific user (authenticated users)"""
    placements = db.query(Placement).filter(Placement.user_id == user_id).all()
    return placements

@app.get("/placements/internship/{internship_id}", response_model=List[PlacementResponse], tags=["Placements"])
async def get_internship_placements(
    internship_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_or_admin)
):
    """Get placements for a specific internship (authenticated users)"""
    placements = db.query(Placement).filter(Placement.internship_id == internship_id).all()
    return placements

# -----------------------
# Waitlist Management
# -----------------------
@app.post("/waitlist/promote", response_model=WaitlistPromotionResponse, tags=["Waitlist"])
async def promote_from_waitlist(
    request: WaitlistPromotionRequest, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Promote next candidate from waitlist for an internship (admin only)"""
    try:
        # If opted out user provided, mark them as opted out
        if request.opted_out_user_id:
            opted_out_placement = db.query(Placement).filter(
                Placement.user_id == request.opted_out_user_id,
                Placement.internship_id == request.internship_id,
                Placement.status == "assigned"
            ).first()
            
            if opted_out_placement:
                opted_out_placement.status = "opted_out"
                db.commit()
                logger.info(f"Marked user {request.opted_out_user_id} as opted out for {request.internship_id}")
            else:
                logger.warning(f"Opted out user {request.opted_out_user_id} not found as assigned for {request.internship_id}")
        
        # Find next waitlist candidate for the internship
        waitlist_placement = db.query(Placement).filter(
            Placement.internship_id == request.internship_id,
            Placement.status == "waitlist"
        ).order_by(Placement.total_score.desc()).first()
        
        if not waitlist_placement:
            return WaitlistPromotionResponse(
                success=False,
                message=f"No waitlist candidates found for internship {request.internship_id}",
                promoted_candidate_id=None
            )
        
        # Check if user is already assigned elsewhere
        existing_assignment = db.query(Placement).filter(
            Placement.user_id == waitlist_placement.user_id,
            Placement.status == "assigned"
        ).first()
        
        if existing_assignment:
            return WaitlistPromotionResponse(
                success=False,
                message=f"User {waitlist_placement.user_id} is already assigned to another internship",
                promoted_candidate_id=None
            )
        
        # Promote the user
        waitlist_placement.status = "assigned"
        db.commit()
        
        return WaitlistPromotionResponse(
            success=True,
            message=f"Successfully promoted user {waitlist_placement.user_id} from waitlist",
            promoted_candidate_id=str(waitlist_placement.user_id)
        )
        
    except Exception as e:
        logger.error(f"Error promoting from waitlist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Waitlist promotion failed: {str(e)}")

# -----------------------
# Health Check
# -----------------------
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedding_model_available": embedding_model is not None,
        "sentence_transformers_available": HAS_ST
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
