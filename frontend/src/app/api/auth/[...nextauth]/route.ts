import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import type { AuthOptions } from "next-auth";

// Fix for static export error
export const dynamic = 'force-dynamic';

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:5000";

// Extend the default session types
declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      name: string;
      email: string;
      image?: string;
    }
  }
  
  interface User {
    id: string;
    name: string;
    email: string;
  }
}

const authOptions: AuthOptions = {
  providers: [
    CredentialsProvider({
      name: "FaceAuth",
      credentials: {
        image: { label: "Image", type: "text" }
      },
      async authorize(credentials) {
        const image = credentials?.image;
        if (!image) return null;

        try {
          // Call backend endpoint directly
          const endpoint = `${BACKEND_URL}/api/face-auth`;
          console.log("🔐 Calling backend endpoint:", endpoint);
          
          const r = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image })
          });

          if (!r.ok) {
            console.error("❌ Backend response not ok:", r.status, r.statusText);
            return null;
          }
          
          const data = await r.json();
          console.log("🔐 Backend response:", data);
          
          if (data?.matched && data?.user) {
            console.log("✅ Face authentication successful for user:", data.user);
            
            // Ensure we have all required user fields
            const user = {
              id: data.user.id || 'unknown',
              name: data.user.name || 'Unknown User',
              email: data.user.email || 'unknown@example.com'
            };
            
            console.log("👤 Returning user object:", user);
            return user;
          }
          
          console.log("❌ Face authentication failed:", data.reason);
          return null;
        } catch (error) {
          console.error("❌ Error during face authentication:", error);
          return null;
        }
      }
    })
  ],
  callbacks: {
    async jwt({ token, user }) {
      // If we have a user, update the token
      if (user) {
        token.id = user.id;
        token.name = user.name;
        token.email = user.email;
        console.log("🔐 JWT callback - updating token with user:", user);
      }
      return token;
    },
    async session({ session, token }) {
      // Update the session with user data from token
      if (token) {
        session.user = {
          ...session.user,
          id: token.id as string,
          name: token.name as string,
          email: token.email as string
        };
        console.log("🔐 Session callback - updating session with token:", token);
      }
      return session;
    }
  },
  session: { strategy: "jwt" as const },
  secret: process.env.NEXTAUTH_SECRET
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST }; 