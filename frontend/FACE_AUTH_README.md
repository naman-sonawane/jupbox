# Face Authentication System - Jupbox Integration

This document explains how to set up and use the face authentication system that's been integrated into your Jupbox app.

## 🚀 Features

- **Face Recognition**: Login using your face instead of passwords
- **User Enrollment**: Register new users with face capture
- **Secure Authentication**: NextAuth integration with JWT sessions
- **MongoDB Storage**: Face embeddings stored securely
- **Roboflow Integration**: AI-powered face detection and embedding generation

## 🛠️ Prerequisites

1. **MongoDB Database** (local or cloud)
2. **Roboflow Account** with a face detection workflow
3. **Node.js 18+** and npm

## 📋 Setup Instructions

### 1. Install Dependencies

The required packages are already installed:
- `next-auth` - Authentication framework
- `mongoose` - MongoDB ODM
- `react-webcam` - Webcam capture component

### 2. Environment Configuration

Copy `env.example` to `.env.local` and configure:

```bash
# MongoDB Connection
MONGODB_URI=mongodb://localhost:27017/jupbox

# NextAuth Configuration
NEXTAUTH_SECRET=your_secure_random_string_here
NEXTAUTH_URL=http://localhost:3000

# Roboflow Configuration
ROBOFLOW_WORKFLOW_URL=https://detect.roboflow.com/your-project/your-version
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

### 3. MongoDB Setup

Start MongoDB locally or use MongoDB Atlas:

```bash
# Local MongoDB
mongod --dbpath /path/to/data/db

# Or use MongoDB Atlas cloud service
```

### 4. Roboflow Workflow

1. Go to [Roboflow](https://roboflow.com)
2. Create a new project for face detection
3. Train a model or use a pre-trained one
4. Deploy as a workflow
5. Copy the workflow URL and API key

## 🔧 How It Works

### Architecture

```
User Face → Webcam → Roboflow → Face Embedding → MongoDB → Authentication
```

### Components

1. **FaceLogin** (`/components/FaceLogin.tsx`)
   - Webcam capture interface
   - Face authentication flow
   - NextAuth integration

2. **FaceEnrollment** (`/components/FaceEnrollment.tsx`)
   - New user registration
   - Face capture and storage
   - Email validation

3. **API Endpoints**
   - `/api/face-auth` - Face verification
   - `/api/enroll` - User enrollment
   - `/api/auth/[...nextauth]` - NextAuth configuration

### Database Schema

```typescript
interface IFaceEmbedding {
  vector: number[];        // CLIP embedding vector
  createdAt?: Date;        // When embedding was created
  note?: string;           // Optional note
}

interface IUser {
  email: string;           // Unique email
  name?: string;           // Optional name
  faceEmbeddings: IFaceEmbedding[]; // Array of face embeddings
}
```

## 🎯 Usage

### User Enrollment

1. Navigate to the Face Enrollment section
2. Enter your email (required) and name (optional)
3. Position your face in the camera
4. Click "Enroll with Face"
5. Wait for processing and confirmation

### User Login

1. Navigate to the Face Login section
2. Position your face in the camera
3. Click "Login with Face"
4. Wait for authentication
5. Success/failure notification

## ⚙️ Configuration

### Threshold Tuning

In `/src/app/api/face-auth/route.ts`:

```typescript
const THRESHOLD = 0.75; // Adjust this value
```

- **Higher values** (0.8-0.9): Stricter matching, fewer false positives
- **Lower values** (0.6-0.7): More lenient matching, more false positives

### Roboflow Field Mapping

Your workflow may output embeddings with different field names. Update the extraction logic:

```typescript
const embedding = (p.embedding ??           // Try 'embedding'
                  (p as any).clip_embedding ?? // Try 'clip_embedding'
                  (p as any).embedding_vector) as number[] | undefined; // Try 'embedding_vector'
```

## 🔒 Security Considerations

### Data Protection

- Face embeddings are stored as numerical vectors
- Original images are not stored
- Embeddings are processed server-side only

### Authentication Flow

- JWT-based sessions
- No password storage
- Secure API endpoints

### Privacy Compliance

- Get user consent for biometric data
- Comply with local biometric laws
- Consider data retention policies

## 🐛 Troubleshooting

### Common Issues

1. **"No face detected"**
   - Ensure good lighting
   - Face should be clearly visible
   - Check camera permissions

2. **"Roboflow error"**
   - Verify API key and workflow URL
   - Check Roboflow service status
   - Validate workflow output format

3. **"MongoDB connection failed"**
   - Check MongoDB service status
   - Verify connection string
   - Ensure network access

4. **"Authentication failed"**
   - Check threshold settings
   - Verify user enrollment
   - Review Roboflow response

### Debug Mode

Enable detailed logging:

```typescript
// In API routes
console.log('Roboflow response:', rfJson);
console.log('Best match score:', bestScore);
console.log('User found:', bestUser);
```

## 🚀 Production Deployment

### Environment Variables

Set production environment variables:

```bash
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/jupbox
NEXTAUTH_SECRET=production_secret_key
NEXTAUTH_URL=https://yourdomain.com
ROBOFLOW_WORKFLOW_URL=https://detect.roboflow.com/prod/version
ROBOFLOW_API_KEY=production_api_key
```

### Scaling Considerations

- **Vector Database**: For many users, consider Pinecone/Qdrant
- **Load Balancing**: Multiple API instances
- **Caching**: Redis for session management
- **Monitoring**: Log aggregation and metrics

## 📚 API Reference

### Face Authentication

```typescript
POST /api/face-auth
Body: { image: string } // base64 encoded image
Response: { matched: boolean, user?: User, score?: number }
```

### User Enrollment

```typescript
POST /api/enroll
Body: { image: string, email: string, name?: string }
Response: { success: boolean, user: User, message: string }
```

### NextAuth Integration

```typescript
// Client-side
import { signIn } from 'next-auth/react';
const result = await signIn('credentials', { image: screenshot });

// Server-side
import { getServerSession } from 'next-auth/next';
const session = await getServerSession(req, res, authOptions);
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License - see main LICENSE file

## 🔗 Resources

- [NextAuth.js Documentation](https://next-auth.js.org/)
- [Mongoose Documentation](https://mongoosejs.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [React Webcam](https://github.com/mozmorris/react-webcam)

---

**Note**: This face authentication system is designed to work alongside your existing Spotify and gesture control features. It provides an additional layer of security and convenience for user authentication. 