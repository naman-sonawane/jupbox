// lib/db.ts
import mongoose from "mongoose";

const MONGODB_URI = process.env.MONGODB_URI;
if (!MONGODB_URI) throw new Error("MONGODB_URI is not set in env");

declare global {
  // allow global cache across hot reloads in dev
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  var _mongoose: { conn: typeof mongoose | null; promise: Promise<typeof mongoose> | null } | undefined;
}

let cached = global._mongoose;

if (!cached) {
  cached = global._mongoose = { conn: null, promise: null };
}

export async function connectDB(): Promise<typeof mongoose> {
  if (cached?.conn) return cached.conn;
  if (!cached?.promise) {
    cached!.promise = mongoose.connect(MONGODB_URI!).then(m => m);
  }
  cached!.conn = await cached!.promise;
  return cached!.conn;
} 