import React from "react";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomePage from "./components/HomePage";
import DiabeticRetinopathyDetector from "./components/DiabeticRetinopathyDetector";

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/detect" element={<DiabeticRetinopathyDetector />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;