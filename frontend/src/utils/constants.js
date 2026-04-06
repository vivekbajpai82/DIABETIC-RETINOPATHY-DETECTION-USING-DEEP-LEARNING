export const SEVERITY_LEVELS = {
  'No_DR': {
    label: 'No Diabetic Retinopathy',
    shortLabel: 'No DR',
    color: '#4caf50',
    bgColor: '#e8f5e8',
    borderColor: '#81c784',
    recommendation: 'No signs of diabetic retinopathy detected. Continue regular eye checkups.',
    urgency: 'low'
  },
  'Mild': {
    label: 'Mild Diabetic Retinopathy',
    shortLabel: 'Mild DR',
    color: '#ff9800',
    bgColor: '#fff3e0',
    borderColor: '#ffb74d',
    recommendation: 'Mild diabetic retinopathy detected. Monitor closely and consult your ophthalmologist.',
    urgency: 'low'
  },
  'Moderate': {
    label: 'Moderate Diabetic Retinopathy',
    shortLabel: 'Moderate DR',
    color: '#ff5722',
    bgColor: '#fff3e0',
    borderColor: '#ff9800',
    recommendation: 'Moderate diabetic retinopathy detected. Schedule an appointment with your eye specialist soon.',
    urgency: 'medium'
  },
  'Severe': {
    label: 'Severe Diabetic Retinopathy',
    shortLabel: 'Severe DR',
    color: '#f44336',
    bgColor: '#ffebee',
    borderColor: '#ef5350',
    recommendation: 'Severe diabetic retinopathy detected. Immediate consultation with an ophthalmologist is recommended.',
    urgency: 'high'
  },
  'Proliferative': {
    label: 'Proliferative Diabetic Retinopathy',
    shortLabel: 'Proliferative DR',
    color: '#9c27b0',
    bgColor: '#f3e5f5',
    borderColor: '#ba68c8',
    recommendation: 'Proliferative diabetic retinopathy detected. Urgent medical attention required.',
    urgency: 'critical'
  }
};

export const API_ENDPOINTS = {
  UPLOAD: '/',
  JSON_PREDICT: '/api/predict',
  IMAGES: '/uploads'
};

export const FILE_CONSTRAINTS = {
  MAX_SIZE: 10 * 1024 * 1024, // 10MB
  ACCEPTED_TYPES: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
  MIN_CONFIDENCE: 50
};

export const UI_MESSAGES = {
  UPLOAD_INSTRUCTIONS: 'Drop your retinal image here or click to browse files',
  PROCESSING: 'Analyzing retinal image...',
  PROCESSING_SUB: 'Please wait while our AI processes the image',
  NO_RESULTS: 'Upload a retinal image to see analysis results',
  NO_RESULTS_SUB: 'The AI model will classify the severity of diabetic retinopathy'
};