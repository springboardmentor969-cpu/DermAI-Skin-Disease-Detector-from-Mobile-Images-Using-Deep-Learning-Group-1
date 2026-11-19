import React, { createContext, useContext, useState, useEffect } from 'react';
import { authApi } from '../api/client';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('dermai_token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initializeAuth = async () => {
      const storedToken = localStorage.getItem('dermai_token');
      if (storedToken) {
        try {
          const profile = await authApi.getMe();
          setUser(profile);
          setToken(storedToken);
        } catch (err) {
          console.error('Failed to restore session:', err);
          localStorage.removeItem('dermai_token');
          localStorage.removeItem('dermai_user');
          setToken(null);
          setUser(null);
        }
      }
      setLoading(false);
    };

    initializeAuth();
  }, []);

  const login = async (email, password) => {
    const data = await authApi.login({ email, password });
    localStorage.setItem('dermai_token', data.access_token);
    setToken(data.access_token);

    const profile = await authApi.getMe();
    setUser(profile);
    localStorage.setItem('dermai_user', JSON.stringify(profile));
    return profile;
  };

  const register = async (userData) => {
    const newUser = await authApi.register(userData);
    // Automatically log in following registration
    const data = await authApi.login({ email: userData.email, password: userData.password });
    localStorage.setItem('dermai_token', data.access_token);
    setToken(data.access_token);
    setUser(newUser);
    localStorage.setItem('dermai_user', JSON.stringify(newUser));
    return newUser;
  };

  const logout = () => {
    localStorage.removeItem('dermai_token');
    localStorage.removeItem('dermai_user');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!token && !!user,
        loading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
