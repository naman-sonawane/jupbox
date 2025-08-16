// models/User.ts
import mongoose, { Document } from "mongoose";

export interface IFaceEmbedding {
  vector: number[];
  createdAt?: Date;
  note?: string;
}

export interface IUser extends Document {
  email: string;
  name?: string;
  faceEmbeddings: IFaceEmbedding[];
}

const FaceEmbeddingSchema = new mongoose.Schema<IFaceEmbedding>({
  vector: { type: [Number], required: true },
  createdAt: { type: Date, default: Date.now },
  note: { type: String }
});

const UserSchema = new mongoose.Schema<IUser>({
  email: { type: String, required: true, unique: true },
  name: { type: String },
  faceEmbeddings: { type: [FaceEmbeddingSchema], default: [] }
});

export default mongoose.models.User || mongoose.model<IUser>("User", UserSchema); 