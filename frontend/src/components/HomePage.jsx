import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, Upload, CheckCircle, ArrowRight, Activity, Shield, Zap } from 'lucide-react';
import './HomePage.css';

export default function HomePage() {
  const [isHovered, setIsHovered] = useState(false);
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/detect');
  };

  return (
    <div className="home-container">
      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-content">
          <div className="nav-logo">
            <Eye size={32} color="#2563eb" />
            <span className="logo-text">DR Detection</span>
          </div>
          <div className="nav-links">
            <a href="#features">Features</a>
            <a href="#about">About</a>
            <a href="#contact">Contact</a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-grid">
          {/* Left Content */}
          <div className="hero-left">
            <div className="badge">
              <span>AI-Powered Detection</span>
            </div>
            
            <h1 className="hero-title">
              Detect <span className="gradient-text">Diabetic Retinopathy</span> Early
            </h1>
            
            <p className="hero-description">
              Advanced AI technology to help detect diabetic retinopathy from retinal images. 
              Fast, accurate, and accessible healthcare screening.
            </p>

            <div className="button-group">
              <button
                onClick={handleGetStarted}
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
                className="btn-primary"
              >
                <Upload size={20} />
                <span>Get Started</span>
                <ArrowRight size={20} className={isHovered ? 'arrow-move' : ''} />
              </button>
              
              <button className="btn-secondary">
                Learn More
              </button>
            </div>

            {/* Stats */}
            <div className="stats-grid">
              <div className="stat-item">
                <div className="stat-value blue">98%</div>
                <div className="stat-label">Accuracy</div>
              </div>
              <div className="stat-item">
                <div className="stat-value purple">50K+</div>
                <div className="stat-label">Scans Done</div>
              </div>
              <div className="stat-item">
                <div className="stat-value green">24/7</div>
                <div className="stat-label">Available</div>
              </div>
            </div>
          </div>

          {/* Right Visual */}
          <div className="hero-right">
            <div className="upload-card">
              <div className="upload-card-inner">
                <div className="upload-header">
                  <div className="icon-box blue-bg">
                    <Eye size={32} color="#2563eb" />
                  </div>
                  <div>
                    <div className="upload-title">Upload Retinal Image</div>
                    <div className="upload-subtitle">Get instant analysis</div>
                  </div>
                </div>

                <div className="upload-area">
                  <Upload size={48} color="#9ca3af" />
                  <div className="upload-text">Click to upload or drag and drop</div>
                  <div className="upload-subtext">PNG, JPG up to 10MB</div>
                </div>

                <div className="upload-footer">
                  <div className="footer-item green">
                    <Shield size={16} />
                    <span>Secure & Private</span>
                  </div>
                  <div className="footer-item blue">
                    <Zap size={16} />
                    <span>Instant Results</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Floating Cards */}
            <div className="floating-card top-right">
              <CheckCircle size={20} color="#10b981" />
              <span>FDA Approved</span>
            </div>
            
            <div className="floating-card bottom-left">
              <Activity size={20} color="#a855f7" />
              <span>Real-time Analysis</span>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div id="features" className="features-section">
          <div className="features-header">
            <h2>Why Choose Our Platform?</h2>
            <p>Advanced technology for better healthcare outcomes</p>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon blue-bg">
                <Zap size={32} color="#2563eb" />
              </div>
              <h3>Fast Detection</h3>
              <p>Get results in seconds with our AI-powered analysis engine.</p>
            </div>

            <div className="feature-card">
              <div className="feature-icon purple-bg">
                <Shield size={32} color="#a855f7" />
              </div>
              <h3>Secure & Private</h3>
              <p>Your data is encrypted and never shared with third parties.</p>
            </div>

            <div className="feature-card">
              <div className="feature-icon green-bg">
                <CheckCircle size={32} color="#10b981" />
              </div>
              <h3>High Accuracy</h3>
              <p>98% accuracy rate validated by medical professionals.</p>
            </div>
          </div>
        </div>

        {/* About Section */}
        <div id="about" className="about-section">
          <div className="about-content">
            <div className="about-text">
              <h2>About Our Platform</h2>
              <p>
                We are dedicated to making diabetic retinopathy detection accessible to everyone. 
                Our AI-powered platform uses advanced deep learning algorithms trained on millions 
                of retinal images to provide accurate, fast, and reliable screening results.
              </p>
              <p>
                Early detection of diabetic retinopathy can prevent vision loss and improve treatment 
                outcomes. Our mission is to democratize healthcare by providing cutting-edge diagnostic 
                tools that are easy to use and accessible worldwide.
              </p>
              <div className="about-stats">
                <div className="about-stat">
                  <h3>5+ Years</h3>
                  <p>Research & Development</p>
                </div>
                <div className="about-stat">
                  <h3>100K+</h3>
                  <p>Patients Screened</p>
                </div>
                <div className="about-stat">
                  <h3>50+</h3>
                  <p>Healthcare Partners</p>
                </div>
              </div>
            </div>
            <div className="about-image">
              <div className="image-placeholder">
                <Eye size={80} color="#2563eb" />
                <p>AI-Powered Analysis</p>
              </div>
            </div>
          </div>
        </div>

        {/* Contact Section */}
        <div id="contact" className="contact-section">
          <div className="contact-header">
            <h2>Get In Touch</h2>
            <p>Have questions? We'd love to hear from you.</p>
          </div>
          <div className="contact-grid">
            <div className="contact-card">
              <div className="contact-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
                </svg>
              </div>
              <h3>Email Us</h3>
              <p>support@drdetection.com</p>
              <p>info@drdetection.com</p>
            </div>
            <div className="contact-card">
              <div className="contact-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72 12.84 12.84 0 00.7 2.81 2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45 12.84 12.84 0 002.81.7A2 2 0 0122 16.92z"/>
                </svg>
              </div>
              <h3>Call Us</h3>
              <p>+1 (555) 123-4567</p>
              <p>Mon-Fri, 9AM-6PM EST</p>
            </div>
            <div className="contact-card">
              <div className="contact-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/>
                  <circle cx="12" cy="10" r="3"/>
                </svg>
              </div>
              <h3>Visit Us</h3>
              <p>123 Healthcare Ave</p>
              <p>Medical District, CA 94102</p>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="cta-section">
          <h2>Ready to Get Started?</h2>
          <p>Upload your first retinal image and get instant results</p>
          <button onClick={handleGetStarted} className="cta-button">
            Start Free Screening
          </button>
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-grid">
            <div className="footer-column">
              <div className="footer-logo">
                <Eye size={24} />
                <span>DR Detection</span>
              </div>
              <p>AI-powered diabetic retinopathy detection platform.</p>
            </div>
            <div className="footer-column">
              <h4>Product</h4>
              <div className="footer-links">
                <div>Features</div>
                <div>Pricing</div>
                <div>API</div>
              </div>
            </div>
            <div className="footer-column">
              <h4>Company</h4>
              <div className="footer-links">
                <div>About</div>
                <div>Blog</div>
                <div>Careers</div>
              </div>
            </div>
            <div className="footer-column">
              <h4>Legal</h4>
              <div className="footer-links">
                <div>Privacy</div>
                <div>Terms</div>
                <div>Security</div>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            © 2025 DR Detection. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}