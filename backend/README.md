# Smart Allocation Engine API

A FastAPI-based smart allocation engine for internship matching with fairness considerations.

## Features

- **Smart Matching Algorithm**: Uses semantic similarity, qualifications, CGPA, location, and fairness boosts
- **Two-Phase Matching**: Reserved seats for beneficiaries, then general allocation
- **Waitlist Management**: Automatic waitlist generation and promotion capabilities
- **Fairness Considerations**: Affirmative action boosts for underrepresented groups
- **PostgreSQL Integration**: Robust database storage for candidates, internships, and placements
- **RESTful API**: Complete CRUD operations for all entities
- **CORS Support**: Ready for Next.js frontend integration
- **Authentication & Authorization**: JWT-based authentication with role-based access control (Admin/User)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Database

Create a PostgreSQL database and update the connection string in your environment:

```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env with your database credentials
DATABASE_URL=postgresql://username:password@localhost/smart_allocation
SECRET_KEY=your-secret-key-change-this-in-production
```

### 3. Initialize Authentication

Set up the database with default admin and user accounts:

```bash
python init_auth.py
```

This creates:
- **Admin user**: `admin` / `admin123` (full access)
- **Regular user**: `user` / `user123` (read-only access)

**⚠️ Important**: Change these default passwords in production!

### 4. Run the Application

```bash
# Development mode with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 6. Test Authentication

Run the authentication test script:

```bash
python test_auth.py
```

## Authentication

The API uses JWT-based authentication with role-based access control:

### User Roles

- **Admin**: Full access to all endpoints (create, read, update, delete)
- **User**: Read-only access to most endpoints

### Authentication Endpoints

- `POST /auth/login` - Login and get access token
- `POST /auth/register` - Register new user (admin only)
- `GET /auth/me` - Get current user information
- `GET /auth/users` - List all users (admin only)
- `PUT /auth/users/{user_id}` - Update user (admin only)

### Using Authentication

1. **Login** to get an access token:
```bash
curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin123"}'
```

2. **Use the token** in subsequent requests:
```bash
curl -X GET "http://localhost:8000/candidates/" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Access Control

| Endpoint | Admin | User | Public |
|----------|-------|------|--------|
| `/auth/login` | ✓ | ✓ | ✓ |
| `/auth/register` | ✓ | ✗ | ✗ |
| `/auth/me` | ✓ | ✓ | ✗ |
| `/auth/users` | ✓ | ✗ | ✗ |
| `/candidates/` (GET) | ✓ | ✓ | ✗ |
| `/candidates/` (POST/PUT/DELETE) | ✓ | ✗ | ✗ |
| `/internships/` (GET) | ✓ | ✓ | ✗ |
| `/internships/` (POST/PUT/DELETE) | ✓ | ✗ | ✗ |
| `/matching/run` | ✓ | ✗ | ✗ |
| `/placements/` (GET) | ✓ | ✓ | ✗ |
| `/waitlist/promote` | ✓ | ✗ | ✗ |
| `/health` | ✓ | ✓ | ✓ |

## API Endpoints

### Candidates
- `POST /candidates/` - Create a new candidate
- `GET /candidates/` - List all candidates
- `GET /candidates/{id}` - Get specific candidate
- `PUT /candidates/{id}` - Update candidate
- `DELETE /candidates/{id}` - Delete candidate

### Internships
- `POST /internships/` - Create a new internship
- `GET /internships/` - List all internships
- `GET /internships/{id}` - Get specific internship
- `PUT /internships/{id}` - Update internship
- `DELETE /internships/{id}` - Delete internship

### Matching
- `POST /matching/run` - Run the allocation algorithm
- `GET /placements/` - Get all placements
- `GET /placements/candidate/{id}` - Get candidate's placements
- `GET /placements/internship/{id}` - Get internship's placements

### Waitlist Management
- `POST /waitlist/promote` - Promote next candidate from waitlist

## Data Models

### Candidate
```json
{
  "id": "string",
  "name": "string",
  "qualification": "string",
  "skills": "string (semicolon-separated)",
  "profile_text": "string",
  "district": "string",
  "category": "string (gen/sc/st/obc/ews)",
  "past_participation": "boolean",
  "cgpa": "float",
  "distance": "float",
  "age": "integer",
  "income": "float",
  "gender": "string",
  "pwd": "boolean"
}
```

### Internship
```json
{
  "id": "string",
  "org": "string",
  "role": "string",
  "required_skills": "string (semicolon-separated)",
  "description": "string",
  "min_qualification": "string",
  "capacity": "integer",
  "district": "string",
  "sector": "string",
  "reserved_percent": "integer (0-100)"
}
```

## Matching Algorithm

The algorithm uses a weighted scoring system with the following components:

1. **Semantic Similarity** (55%): Profile text similarity using sentence transformers
2. **Qualification Match** (15%): Educational qualification compatibility
3. **CGPA Score** (15%): Academic performance percentile
4. **Location Preference** (8%): Geographic proximity bonus
5. **Past Participation Penalty** (5%): Reduces score for previous participants
6. **Affirmative Action Boost** (2%): Fairness considerations

### Two-Phase Matching

1. **Phase 1**: Beneficiaries (SC/ST/rural/aspirational districts) get reserved seats
2. **Phase 2**: Remaining seats allocated to all eligible candidates

### Fairness Features

- **Category-based boosts**: SC/ST (5%), OBC/EWS (3%)
- **Geographic boosts**: Aspirational districts (3%), Rural areas (2%)
- **Income-based boosts**: Low income families get additional points
- **Gender diversity**: Female candidates get small boost
- **Accessibility**: PWD candidates get additional consideration

## Integration with Next.js

The API is designed to work seamlessly with Next.js frontends:

```javascript
// Example API call from Next.js
const response = await fetch('http://localhost:8000/matching/run', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    aspirational_districts: ['district1', 'district2'],
    beneficiary_categories: ['sc', 'st'],
    age_min: 21,
    age_max: 30,
    use_embeddings: true
  })
});

const result = await response.json();
```

## Development

### Database Migrations

The application uses SQLAlchemy with automatic table creation. For production, consider using Alembic for migrations:

```bash
# Initialize Alembic (if needed)
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Initial migration"

# Apply migration
alembic upgrade head
```

### Testing

```bash
# Run with test database
DATABASE_URL=sqlite:///./test.db uvicorn app:app --reload
```

## Production Deployment

1. **Environment Variables**: Set production database URL and security keys
2. **CORS Configuration**: Update allowed origins for your domain
3. **Database**: Use PostgreSQL with proper connection pooling
4. **Reverse Proxy**: Use Nginx or similar for production
5. **SSL**: Enable HTTPS for secure communication

## License

This project is part of the Smart India Hackathon (SIH) initiative.
