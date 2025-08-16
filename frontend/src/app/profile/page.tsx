'use client';

import { useState, useEffect } from 'react';
import { useSession, signOut } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function ProfilePage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [userInfo, setUserInfo] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Check authentication
    if (status === 'unauthenticated') {
      router.push('/login');
      return;
    }

    // Fetch user profile data
    if (status === 'authenticated') {
      fetchUserProfile();
    }
  }, [status, router]);

  const fetchUserProfile = async () => {
    try {
      setLoading(true);
      // You can fetch additional user data from your backend here
      // For now, we'll use the session data
      setUserInfo({
        name: session?.user?.name || 'User',
        email: session?.user?.email || 'user@example.com',
        image: session?.user?.image,
        joinedDate: new Date().toLocaleDateString(),
        lastLogin: new Date().toLocaleDateString(),
        faceEnrolled: true,
        totalLogins: 1
      });
    } catch (error) {
      console.error('Error fetching user profile:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    setLoading(true);
    try {
      await signOut({ redirect: false });
      router.push('/login');
    } catch (error) {
      console.error('Logout error:', error);
      setLoading(false);
    }
  };

  if (status === 'loading' || loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-black to-purple-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-purple-400 mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-black to-purple-900">
      {/* Navigation Bar */}
      <nav className="bg-black/20 backdrop-blur-md border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <Link href="/" className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
                  <span className="text-white text-xl font-bold">J</span>
                </div>
                <span className="text-xl font-bold text-white">Jupbox</span>
              </Link>
            </div>
            
            <div className="flex items-center space-x-6">
              <Link 
                href="/spotify" 
                className="text-gray-300 hover:text-white transition-colors font-medium"
              >
                🎵 Spotify
              </Link>
              <Link 
                href="/emotions" 
                className="text-gray-300 hover:text-white transition-colors font-medium"
              >
                🎭 Emotions
              </Link>
              <Link 
                href="/profile" 
                className="text-purple-300 hover:text-white transition-colors font-medium"
              >
                👤 Profile
              </Link>
              <button
                onClick={handleLogout}
                disabled={loading}
                className="px-4 py-2 bg-red-500/20 text-red-300 rounded-lg hover:bg-red-500/30 transition-colors border border-red-500/30 disabled:opacity-50"
              >
                {loading ? '🚪 Logging out...' : '🚪 Logout'}
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Profile Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-8"
        >
          <div className="w-32 h-32 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-6">
            {userInfo?.image ? (
              <img 
                src={userInfo.image} 
                alt="Profile" 
                className="w-full h-full rounded-full object-cover"
              />
            ) : (
              <span className="text-4xl">👤</span>
            )}
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">{userInfo?.name}</h1>
          <p className="text-xl text-gray-300">{userInfo?.email}</p>
        </motion.div>

        {/* Profile Information */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20 mb-8"
        >
          <h2 className="text-2xl font-bold text-white mb-6">👤 Profile Information</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Full Name</label>
                <div className="px-4 py-3 bg-white/10 border border-purple-500/30 rounded-xl text-white">
                  {userInfo?.name}
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Email Address</label>
                <div className="px-4 py-3 bg-white/10 border border-purple-500/30 rounded-xl text-white">
                  {userInfo?.email}
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Member Since</label>
                <div className="px-4 py-3 bg-white/10 border border-purple-500/30 rounded-xl text-white">
                  {userInfo?.joinedDate}
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Last Login</label>
                <div className="px-4 py-3 bg-white/10 border border-purple-500/30 rounded-xl text-white">
                  {userInfo?.lastLogin}
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Account Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20 mb-8"
        >
          <h2 className="text-2xl font-bold text-white mb-6">🔐 Account Status</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-6 bg-green-500/20 rounded-xl border border-green-500/30">
              <div className="text-3xl mb-2">✅</div>
              <h3 className="text-lg font-semibold text-white mb-2">Face Authentication</h3>
              <p className="text-green-300 text-sm">Enrolled & Active</p>
            </div>
            
            <div className="text-center p-6 bg-blue-500/20 rounded-xl border border-blue-500/30">
              <div className="text-3xl mb-2">🎵</div>
              <h3 className="text-lg font-semibold text-white mb-2">Spotify Connected</h3>
              <p className="text-blue-300 text-sm">Ready to use</p>
            </div>
            
            <div className="text-center p-6 bg-purple-500/20 rounded-xl border border-purple-500/30">
              <div className="text-3xl mb-2">🎭</div>
              <h3 className="text-lg font-semibold text-white mb-2">Emotion Detection</h3>
              <p className="text-purple-300 text-sm">Available</p>
            </div>
          </div>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20 mb-8"
        >
          <h2 className="text-2xl font-bold text-white mb-6">⚡ Quick Actions</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Link 
              href="/spotify"
              className="p-6 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl border border-purple-500/30 hover:border-purple-500/50 transition-all duration-300 hover:scale-105 group"
            >
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                  <span className="text-2xl">🎵</span>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">Go to Spotify</h3>
                  <p className="text-gray-300 text-sm">Control your music</p>
                </div>
              </div>
            </Link>
            
            <Link 
              href="/emotions"
              className="p-6 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl border border-purple-500/30 hover:border-purple-500/50 transition-all duration-300 hover:scale-105 group"
            >
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                  <span className="text-2xl">🎭</span>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">Emotion Analysis</h3>
                  <p className="text-gray-300 text-sm">Check your mood</p>
                </div>
              </div>
            </Link>
          </div>
        </motion.div>

        {/* Account Management */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20"
        >
          <h2 className="text-2xl font-bold text-white mb-6">⚙️ Account Management</h2>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-xl">
              <div>
                <h3 className="text-lg font-semibold text-white">Face Recognition</h3>
                <p className="text-gray-300 text-sm">Update your facial enrollment</p>
              </div>
              <button className="px-4 py-2 bg-purple-500/20 text-purple-300 rounded-lg hover:bg-purple-500/30 transition-colors border border-purple-500/30">
                Update
              </button>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-xl">
              <div>
                <h3 className="text-lg font-semibold text-white">Password</h3>
                <p className="text-gray-300 text-sm">Change your account password</p>
              </div>
              <button className="px-4 py-2 bg-purple-500/20 text-purple-300 rounded-lg hover:bg-purple-500/30 transition-colors border border-purple-500/30">
                Change
              </button>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-red-500/20 rounded-xl border border-red-500/30">
              <div>
                <h3 className="text-lg font-semibold text-white">Delete Account</h3>
                <p className="text-red-300 text-sm">Permanently remove your account</p>
              </div>
              <button className="px-4 py-2 bg-red-500/20 text-red-300 rounded-lg hover:bg-red-500/30 transition-colors border border-red-500/30">
                Delete
              </button>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
