import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    console.log("Received enrollment request:", { 
      hasFrames: !!body.frames, 
      hasEmail: !!body.email, 
      hasName: !!body.name,
      frameCount: body.frames?.length || 0
    });
    
    const { frames, email, name } = body;
    
    if (!frames || !email || !Array.isArray(frames) || frames.length === 0) {
      console.log("Missing required fields:", { frames: !!frames, email: !!email, frameCount: frames?.length });
      return NextResponse.json({ 
        error: "Missing required fields: frames array and email" 
      }, { status: 400 });
    }
    
    try {
      console.log("Calling backend enrollment API...");
      const backendResp = await fetch(`${BACKEND_URL}/api/enroll`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          frames, 
          email, 
          name
        })
      });
      
      console.log("Backend response status:", backendResp.status);
      
      if (!backendResp.ok) {
        const text = await backendResp.text();
        console.error("Backend error:", text);
        return NextResponse.json({ error: "Enrollment failed", details: text }, { status: 502 });
      }
      
      const backendData = await backendResp.json();
      console.log("Backend response data:", backendData);
      
      if (backendData.success) {
        return NextResponse.json({
          success: true,
          user_id: backendData.user_id,
          inserted_points: backendData.inserted_points,
          message: backendData.message
        });
      } else {
        return NextResponse.json({ 
          error: backendData.error || "Enrollment failed" 
        }, { status: 400 });
      }
      
    } catch (error) {
      console.error("Backend API call failed:", error);
      return NextResponse.json({ 
        error: "Failed to connect to backend service" 
      }, { status: 502 });
    }
    
  } catch (error) {
    console.error("Enrollment API error:", error);
    return NextResponse.json({ 
      error: "Internal server error" 
    }, { status: 500 });
  }
} 