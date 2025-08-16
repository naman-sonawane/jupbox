import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    console.log("Received face authentication request:", { 
      hasImage: !!body.image, 
      imageLength: body.image?.length || 0
    });
    
    const { image } = body;
    
    if (!image) {
      console.log("No image provided");
      return NextResponse.json({ matched: false, reason: "no_image_provided" });
    }
    
    try {
      console.log("Calling backend face authentication API...");
      const backendResp = await fetch(`${BACKEND_URL}/api/face-auth`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image })
      });
      
      console.log("Backend response status:", backendResp.status);
      
      if (!backendResp.ok) {
        const text = await backendResp.text();
        console.error("Backend error:", text);
        return NextResponse.json({ matched: false, reason: "backend_error", details: text });
      }
      
      const backendData = await backendResp.json();
      console.log("Backend response data:", backendData);
      
      // Return the backend response directly
      return NextResponse.json(backendData);
      
    } catch (error) {
      console.error("Backend API call failed:", error);
      return NextResponse.json({ 
        matched: false, 
        reason: "backend_unavailable" 
      });
    }
    
  } catch (error) {
    console.error("Face authentication API error:", error);
    return NextResponse.json({ 
      matched: false, 
      reason: "internal_error" 
    });
  }
} 