import React, { useState } from 'react';
import './App.css';

function App() {
  const [files, setFiles] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedType, setSelectedType] = useState('auto');

  const handleFileSelect = (event) => {
    const selectedFiles = Array.from(event.target.files);
    setFiles(selectedFiles);
  };

 const handlePredict = async () => {  // ← ADDED 'async' HERE!
  if (files.length === 0) {
    alert('Please select at least one image');
    return;
  }

  setLoading(true);
  setPredictions([]);

  try {
    const endpoint = selectedType === 'batch' 
      ? 'http://127.0.0.1:8000/predict/batch'
      : selectedType === 'leaf'
      ? 'http://127.0.0.1:8000/predict/leaf'
      : selectedType === 'fruit'
      ? 'http://127.0.0.1:8000/predict/fruit'
      : 'http://127.0.0.1:8000/predict/auto';

    const formData = new FormData();

    // 🔥 CRITICAL FIX: Different field names for batch vs single
    if (selectedType === 'batch') {
      // Batch endpoint expects 'files' (plural)
      files.forEach(file => {
        formData.append('files', file);
      });
    } else {
      // Single image endpoints expect 'file' (singular)
      formData.append('file', files[0]);
    }

    console.log(`Sending to: ${endpoint}`);
    console.log(`Field name: ${selectedType === 'batch' ? 'files' : 'file'}`);
    console.log(`Files: ${files.length}`);

    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
      // NO headers needed for FormData!
    });

    console.log(`Response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Backend error: ${errorText}`);
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    console.log('Response data:', data);

    if (selectedType === 'batch') {
      setPredictions(data.predictions || []);
    } else {
      setPredictions([{ ...data, filename: files[0].name }]);
    }
  } catch (error) {
    console.error('Error details:', error);
    alert(`Prediction failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
};

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'bg-red-100 text-red-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🍅 TomatoGuard
          </h1>
          <p className="text-gray-600">
            AI-powered tomato leaf and fruit disease detection
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column: Upload & Controls */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Upload Images
              </h2>
              
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Detection Type
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {['auto', 'leaf', 'fruit', 'batch'].map((type) => (
                    <button
                      key={type}
                      onClick={() => setSelectedType(type)}
                      className={`px-4 py-2 rounded-lg transition-colors ${
                        selectedType === type
                          ? 'bg-green-600 text-white'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </button>
                  ))}
                </div>
                <p className="text-sm text-gray-500 mt-2">
                  {selectedType === 'auto' && 'Auto-detect leaf or fruit'}
                  {selectedType === 'leaf' && 'Force leaf detection'}
                  {selectedType === 'fruit' && 'Force fruit detection'}
                  {selectedType === 'batch' && 'Batch process multiple images'}
                </p>
              </div>

              <div className="mb-6">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-green-500 transition-colors">
                  <input
                    type="file"
                    id="file-upload"
                    multiple={selectedType === 'batch'}
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer flex flex-col items-center"
                  >
                    <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mb-4">
                      <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                    </div>
                    <span className="text-gray-700 font-medium">
                      {files.length > 0 
                        ? `${files.length} file(s) selected`
                        : 'Choose images'
                      }
                    </span>
                    <span className="text-sm text-gray-500 mt-1">
                      JPG, PNG, or BMP • Max 10MB each
                    </span>
                  </label>
                </div>

                {files.length > 0 && (
                  <div className="mt-4 space-y-2">
                    {files.slice(0, 3).map((file, index) => (
                      <div key={index} className="flex items-center text-sm text-gray-600">
                        <svg className="w-4 h-4 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {file.name}
                      </div>
                    ))}
                    {files.length > 3 && (
                      <div className="text-sm text-gray-500">
                        + {files.length - 3} more files
                      </div>
                    )}
                  </div>
                )}
              </div>

              <button
                onClick={handlePredict}
                disabled={loading || files.length === 0}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  'Detect Diseases'
                )}
              </button>
            </div>

            {/* Model Info */}
            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Model Information
              </h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Model:</span>
                  <span className="font-medium">EfficientNetB0</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Input Size:</span>
                  <span className="font-medium">224×224 pixels</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Diseases:</span>
                  <span className="font-medium">8 total</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-6">
                Detection Results
              </h2>

              {predictions.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <p className="text-gray-500">
                    {loading 
                      ? 'Analyzing images...' 
                      : 'Upload images and click "Detect Diseases" to get predictions'
                    }
                  </p>
                </div>
              ) : (
                <div className="space-y-8">
                  {predictions.map((prediction, index) => (
                    <div key={index} className="border rounded-lg p-6">
                      {/* File info */}
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h3 className="font-medium text-gray-900">
                            {prediction.filename || `Image ${index + 1}`}
                          </h3>
                          <div className="flex items-center space-x-4 mt-1">
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(prediction.severity)}`}>
                              {prediction.severity || 'Unknown'}
                            </span>
                            <span className="text-sm text-gray-500">
                              {prediction.type === 'leaf' ? '🍃 Leaf' : '🍅 Fruit'}
                            </span>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-2xl font-bold text-green-600">
                            {(prediction.confidence * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-gray-500">Confidence</div>
                        </div>
                      </div>

                      {/* Disease info */}
                      <div className="mb-6">
                        <h4 className="text-lg font-semibold text-gray-800 mb-2">
                          {prediction.display_name || prediction.disease}
                        </h4>
                        <p className="text-gray-600">
                          Detected {prediction.type} disease with high confidence
                        </p>
                      </div>

                      {/* Recommendations */}
                      <div className="space-y-6">
                        <div>
                          <h5 className="font-medium text-gray-800 mb-3 flex items-center">
                            <svg className="w-5 h-5 text-green-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            Immediate Recommendations
                          </h5>
                          <ul className="space-y-2">
                            {prediction.recommendations?.map((rec, i) => (
                              <li key={i} className="flex items-start">
                                <span className="inline-block w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0" />
                                <span className="text-gray-700">{rec}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div>
                          <h5 className="font-medium text-gray-800 mb-3 flex items-center">
                            <svg className="w-5 h-5 text-blue-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                            </svg>
                            Prevention Tips
                          </h5>
                          <ul className="space-y-2">
                            {prediction.prevention?.map((prev, i) => (
                              <li key={i} className="flex items-start">
                                <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0" />
                                <span className="text-gray-700">{prev}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        {prediction.organic_control && (
                          <div>
                            <h5 className="font-medium text-gray-800 mb-3 flex items-center">
                              <svg className="w-5 h-5 text-yellow-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                              </svg>
                              Organic Control Methods
                            </h5>
                            <ul className="space-y-2">
                              {prediction.organic_control.map((org, i) => (
                                <li key={i} className="flex items-start">
                                  <span className="inline-block w-2 h-2 bg-yellow-500 rounded-full mt-2 mr-3 flex-shrink-0" />
                                  <span className="text-gray-700">{org}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>

                      {/* Confidence breakdown */}
                      {prediction.all_probabilities && (
                        <div className="mt-6 pt-6 border-t">
                          <h5 className="font-medium text-gray-800 mb-3">
                            Confidence Breakdown
                          </h5>
                          <div className="space-y-2">
                            {Object.entries(prediction.all_probabilities)
                              .sort(([, a], [, b]) => b - a)
                              .map(([disease, prob]) => (
                                <div key={disease} className="flex items-center">
                                  <div className="w-32 text-sm text-gray-600 truncate">
                                    {disease.replace(/_/g, ' ')}
                                  </div>
                                  <div className="flex-1 ml-4">
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                      <div
                                        className="bg-green-600 h-2 rounded-full"
                                        style={{ width: `${prob * 100}%` }}
                                      />
                                    </div>
                                  </div>
                                  <div className="w-16 text-right text-sm font-medium">
                                    {(prob * 100).toFixed(1)}%
                                  </div>
                                </div>
                              ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>TomatoGuard v1.0 • Built with EfficientNetB0 • For agricultural research purposes</p>
        </footer>
      </div>
    </div>
  );
}

export default App;