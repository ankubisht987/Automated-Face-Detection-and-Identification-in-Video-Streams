import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const Dashboard = () => {
    const [imageFile, setImageFile] = useState(null);
    const [videoFile, setVideoFile] = useState(null);
    const [previewImageUrl, setPreviewImageUrl] = useState('');
    const [previewVideoUrl, setPreviewVideoUrl] = useState('');
    const [referenceSet, setReferenceSet] = useState(false);
    const [sampleRate, setSampleRate] = useState(5);
    const [similarity, setSimilarity] = useState(0.6);
    const [processing, setProcessing] = useState(false);
    const [message, setMessage] = useState('');
    const [messageType, setMessageType] = useState('info'); // 'info', 'success', 'error', 'warning'
    const [sessionId, setSessionId] = useState(null);
    const [results, setResults] = useState(null);
    const [processingProgress, setProcessingProgress] = useState(0);
    const [gapThreshold, setGapThreshold] = useState(1.0);
    const [segments, setSegments] = useState([]);

    const imageInputRef = useRef(null);
    const videoInputRef = useRef(null);
    const pollIntervalRef = useRef(null);

    // Initialize session on mount

    // 1. Define helper function first
    const showMessage = (text, type = 'info') => {
        setMessage(text);
        setMessageType(type);
        setTimeout(() => setMessage(''), 5000);
    };

    const createSession = useCallback(async () => {
    try {
        const response = await axios.post(`${API_BASE_URL}/create-session`);
        setSessionId(response.data.session_id);
        showMessage('Session created', 'success');
    } catch (error) {
        showMessage('Failed to create session', 'error');
        console.error(error);
    }
}, []);

useEffect(() => {
        createSession();
        return () => {
            if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
            }
            if (previewImageUrl) {
                URL.revokeObjectURL(previewImageUrl);
            }
            if (previewVideoUrl) {
                URL.revokeObjectURL(previewVideoUrl);
            }
        };
    }, [createSession, previewImageUrl, previewVideoUrl]);

    const onImageChange = (e) => {
        const file = e.target.files?.[0] || null;
        setImageFile(file);
        setReferenceSet(false);
        setMessage('');
        if (previewImageUrl) {
            URL.revokeObjectURL(previewImageUrl);
        }
        setPreviewImageUrl(file ? URL.createObjectURL(file) : '');
    };

    const onVideoChange = (e) => {
        const file = e.target.files?.[0] || null;
        setVideoFile(file);
        setMessage('');
        if (previewVideoUrl) {
            URL.revokeObjectURL(previewVideoUrl);
        }
        setPreviewVideoUrl(file ? URL.createObjectURL(file) : '');
    };

    const handleSetReference = async () => {
        if (!imageFile) {
            showMessage('Please choose a reference image first.', 'warning');
            return;
        }

        if (!sessionId) {
            showMessage('Session not initialized', 'error');
            return;
        }

        try {
            setProcessing(true);
            const formData = new FormData();
            formData.append('file', imageFile);

            const response = await axios.post(
                `${API_BASE_URL}/set-reference/${sessionId}`,
                formData,
                {
                    headers: { 'Content-Type': 'multipart/form-data' },
                }
            );

            setReferenceSet(true);
            showMessage(response.data.message || 'Reference face set successfully!', 'success');
        } catch (error) {
            const errorMsg = error.response?.data?.detail || 'Failed to set reference face';
            showMessage(errorMsg, 'error');
            console.error(error);
        } finally {
            setProcessing(false);
        }
    };

    const pollSessionStatus = async (sid) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/session/${sid}/status`);
            const status = response.data.status;

            if (status === 'processing') {
                setProcessingProgress((current) => Math.min(100, current + Math.random() * 20));
            } else if (status === 'completed') {
                setProcessingProgress(100);
                if (pollIntervalRef.current) {
                    clearInterval(pollIntervalRef.current);
                    pollIntervalRef.current = null;
                }
                fetchResults(sid);
            } else if (status === 'error') {
                if (pollIntervalRef.current) {
                    clearInterval(pollIntervalRef.current);
                    pollIntervalRef.current = null;
                }
                showMessage('Video processing failed', 'error');
                setProcessing(false);
            }
        } catch (error) {
            console.error('Poll error:', error);
        }
    };

    const fetchResults = async (sid) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/session/${sid}/results`);
            setResults(response.data);
            showMessage('Processing completed!', 'success');
            setProcessing(false);

            // Fetch segments
            await fetchSegments(sid);
        } catch (error) {
            showMessage('Failed to fetch results', 'error');
            console.error(error);
            setProcessing(false);
        }
    };

    const fetchSegments = async (sid) => {
        try {
            const response = await axios.post(`${API_BASE_URL}/session/${sid}/segments`, {
                gap_threshold: gapThreshold,
            });
            setSegments(response.data.segments || []);
        } catch (error) {
            console.error('Failed to fetch segments:', error);
        }
    };

    const handleStartProcessing = async () => {
        if (!imageFile || !videoFile) {
            showMessage('Please upload both files before starting.', 'warning');
            return;
        }
        if (!referenceSet) {
            showMessage('Please set the reference face first.', 'warning');
            return;
        }
        if (!sessionId) {
            showMessage('Session not initialized', 'error');
            return;
        }

        try {
            setProcessing(true);
            setProcessingProgress(0);
            setResults(null);
            setSegments([]);

            const formData = new FormData();
            formData.append('file', videoFile);

            const response = await axios.post(
                `${API_BASE_URL}/process-video/${sessionId}`,
                formData,
                {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    params: {
                        sample_rate: sampleRate,
                        similarity_threshold: similarity,
                    },
                }
            );

            showMessage(response.data.message || 'Processing started...', 'info');

            // Start polling
            if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
            }
            pollIntervalRef.current = setInterval(() => {
                pollSessionStatus(sessionId);
            }, 2000); // Poll every 2 seconds

        } catch (error) {
            const errorMsg = error.response?.data?.detail || 'Failed to start processing';
            showMessage(errorMsg, 'error');
            setProcessing(false);
            console.error(error);
        }
    };

    const handleDownloadCSV = () => {
        if (!results || !results.detections) {
            showMessage('No detections to download', 'warning');
            return;
        }

        const csv = [
            ['Time', 'Timestamp (s)', 'Frame Number', 'Similarity'],
            ...results.detections.map(d => [
                d.time_string,
                d.timestamp.toFixed(2),
                d.frame_number,
                d.similarity.toFixed(4)
            ])
        ]
            .map(row => row.join(','))
            .join('\n');

        downloadFile(csv, 'face_detections.csv', 'text/csv');
    };

    const handleDownloadSegmentsCSV = () => {
        if (segments.length === 0) {
            showMessage('No segments to download', 'warning');
            return;
        }

        const csv = [
            ['Start Time', 'End Time', 'Duration'],
            ...segments.map(s => [s.start_time, s.end_time, s.duration])
        ]
            .map(row => row.join(','))
            .join('\n');

        downloadFile(csv, 'face_detection_segments.csv', 'text/csv');
    };

    const downloadFile = (content, filename, type) => {
        const element = document.createElement('a');
        element.setAttribute('href', `data:${type};charset=utf-8,${encodeURIComponent(content)}`);
        element.setAttribute('download', filename);
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    };

    const resetAll = async () => {
        try {
            if (sessionId) {
                await axios.post(`${API_BASE_URL}/session/${sessionId}/cleanup`);
            }
            setImageFile(null);
            setVideoFile(null);
            setReferenceSet(false);
            setSampleRate(5);
            setSimilarity(0.6);
            setMessage('');
            setResults(null);
            setSegments([]);
            setProcessingProgress(0);
            if (imageInputRef.current) imageInputRef.current.value = '';
            if (videoInputRef.current) videoInputRef.current.value = '';

            // Create new session
            await createSession();
        } catch (error) {
            console.error('Error during reset:', error);
            await createSession();
        }
    };

    const bothUploaded = Boolean(imageFile && videoFile);

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <h1 className="text-center mt-8 text-5xl font-bold text-blue-700">
                    👤 Face Detection & Recognition App
                </h1>

                <p className="mt-6 text-center text-lg text-gray-700 max-w-2xl mx-auto">
                    This app detects and recognizes faces in videos using RetinaFace for detection
                    and ArcFace for recognition. Upload a reference face image and a video file to get started.
                </p>

                {/* Message Display */}
                {message && (
                    <div
                        className={`mt-6 p-4 rounded-lg font-medium text-white ${
                            messageType === 'success'
                                ? 'bg-green-500'
                                : messageType === 'error'
                                ? 'bg-red-500'
                                : messageType === 'warning'
                                ? 'bg-yellow-500'
                                : 'bg-blue-500'
                        }`}
                    >
                        {message}
                    </div>
                )}

                {/* Upload Section */}
                <h2 className="mt-8 text-3xl font-bold text-gray-800">📁 Upload Files</h2>

                <div className="flex flex-col md:flex-row md:justify-between gap-6 mt-4">
                    {/* Reference Image Upload */}
                    <section className="flex-1 bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition">
                        <h3 className="text-2xl font-semibold text-gray-800">Reference Face Image</h3>
                        <p className="text-sm text-gray-600 mt-1">Upload a reference face image (JPG, PNG). Max 200MB.</p>
                        <div className="mt-4 flex items-center gap-3">
                            <input
                                ref={imageInputRef}
                                type="file"
                                accept="image/jpeg,image/png"
                                onChange={onImageChange}
                                className="hidden"
                                id="image-input"
                                disabled={processing}
                            />
                            <button
                                type="button"
                                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition disabled:opacity-50"
                                onClick={() => imageInputRef.current?.click()}
                                disabled={processing}
                            >
                                Choose Image
                            </button>
                            <span className="text-sm text-gray-600">
                                {imageFile ? imageFile.name : 'JPG, JPEG, PNG • up to 200MB'}
                            </span>
                        </div>
                        {previewImageUrl && (
                            <img
                                src={previewImageUrl}
                                alt="preview"
                                className="mt-4 max-h-48 rounded-lg border border-gray-200"
                            />
                        )}
                    </section>

                    {/* Video Upload */}
                    <section className="flex-1 bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition">
                        <h3 className="text-2xl font-semibold text-gray-800">Video File</h3>
                        <p className="text-sm text-gray-600 mt-1">Upload a video file (MP4). Max 200MB.</p>
                        <div className="mt-4 flex items-center gap-3">
                            <input
                                ref={videoInputRef}
                                type="file"
                                accept="video/mp4,video/mpeg"
                                onChange={onVideoChange}
                                className="hidden"
                                id="video-input"
                                disabled={processing}
                            />
                            <button
                                type="button"
                                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition disabled:opacity-50"
                                onClick={() => videoInputRef.current?.click()}
                                disabled={processing}
                            >
                                Choose Video
                            </button>
                            <span className="text-sm text-gray-600">
                                {videoFile ? videoFile.name : 'MP4, MPEG4 • up to 200MB'}
                            </span>
                        </div>
                        {previewVideoUrl && (
                            <video
                                controls
                                className="mt-4 max-h-48 rounded-lg border border-gray-200"
                            >
                                <source src={previewVideoUrl} />
                                Your browser does not support the video tag.
                            </video>
                        )}
                    </section>
                </div>

                {/* Info Message */}
                {!bothUploaded && (
                    <div className="mt-6 bg-gray-800 text-white p-4 rounded-lg shadow-md text-center font-medium">
                        Please upload both a reference face image and a video file to begin.
                    </div>
                )}

                {/* Control Section */}
                {bothUploaded && (
                    <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-2xl font-semibold text-gray-800">⚙️ Controls</h3>
                            <div className="flex gap-2">
                                <button
                                    className={`px-4 py-2 rounded-lg font-medium transition ${
                                        referenceSet
                                            ? 'bg-green-500 hover:bg-green-600 text-white'
                                            : 'bg-gray-800 hover:bg-gray-900 text-white'
                                    } disabled:opacity-50`}
                                    onClick={handleSetReference}
                                    disabled={referenceSet || processing}
                                >
                                    {referenceSet ? '✓ Reference Set' : 'Set as Reference Face'}
                                </button>
                                <button
                                    className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg font-medium transition disabled:opacity-50"
                                    onClick={resetAll}
                                    disabled={processing}
                                >
                                    Reset
                                </button>
                            </div>
                        </div>

                        {/* Processing Section */}
                        {referenceSet && (
                            <div className="mt-6 border-t pt-6">
                                <h4 className="text-xl font-semibold text-gray-800 mb-4">🔍 Process Video</h4>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {/* Sample Rate */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700">
                                            Sample Rate (process every Nth frame)
                                        </label>
                                        <select
                                            value={sampleRate}
                                            onChange={(e) => setSampleRate(Number(e.target.value))}
                                            className="mt-2 p-2 border border-gray-300 rounded-lg w-full bg-white text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                            disabled={processing}
                                        >
                                            <option value={1}>1 (every frame - slower)</option>
                                            <option value={2}>2</option>
                                            <option value={5}>5 (recommended)</option>
                                            <option value={10}>10</option>
                                            <option value={15}>15</option>
                                            <option value={30}>30 (faster)</option>
                                        </select>
                                        <p className="mt-2 text-sm text-gray-600">Selected sample rate: {sampleRate}</p>
                                    </div>

                                    {/* Similarity Threshold */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700">
                                            Similarity Threshold: {(similarity * 100).toFixed(0)}%
                                        </label>
                                        <input
                                            type="range"
                                            min={0.5}
                                            max={0.95}
                                            step={0.05}
                                            value={similarity}
                                            onChange={(e) => setSimilarity(Number(e.target.value))}
                                            className="mt-2 w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                            disabled={processing}
                                        />
                                        <p className="text-xs text-gray-500 mt-1">Higher = stricter matching</p>
                                    </div>
                                </div>

                                {/* Progress Bar */}
                                {processing && (
                                    <div className="mt-6">
                                        <div className="flex justify-between mb-2">
                                            <span className="text-sm font-medium text-gray-700">Processing Progress</span>
                                            <span className="text-sm font-medium text-gray-700">{Math.round(processingProgress)}%</span>
                                        </div>
                                        <div className="w-full bg-gray-200 rounded-full h-3">
                                            <div
                                                className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                                                style={{ width: `${processingProgress}%` }}
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* Start Button */}
                                <div className="mt-6">
                                    <button
                                        className="px-8 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-lg font-bold text-lg transition transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                                        onClick={handleStartProcessing}
                                        disabled={processing}
                                    >
                                        {processing ? '⏳ Processing...' : '🚀 Start Processing'}
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Results Section */}
                {results && (
                    <div className="mt-8 bg-white p-6 rounded-lg shadow-md">
                        <h2 className="text-3xl font-bold text-gray-800 mb-6">📊 Detection Results</h2>

                        {/* Summary Stats */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                            <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
                                <p className="text-gray-600 text-sm font-medium">Total Detections</p>
                                <p className="text-3xl font-bold text-blue-600 mt-2">
                                    {results.total_detections}
                                </p>
                            </div>
                            {results.detections.length > 0 && (
                                <>
                                    <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
                                        <p className="text-gray-600 text-sm font-medium">First Detection</p>
                                        <p className="text-3xl font-bold text-green-600 mt-2">
                                            {results.detections[0].time_string}
                                        </p>
                                    </div>
                                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
                                        <p className="text-gray-600 text-sm font-medium">Last Detection</p>
                                        <p className="text-3xl font-bold text-purple-600 mt-2">
                                            {results.detections[results.detections.length - 1].time_string}
                                        </p>
                                    </div>
                                </>
                            )}
                        </div>

                        {/* Detections Table */}
                        {results.detections.length > 0 && (
                            <>
                                <h3 className="text-xl font-semibold text-gray-800 mb-4">⏰ Detection Timestamps</h3>
                                <div className="overflow-x-auto mb-4">
                                    <table className="w-full border-collapse">
                                        <thead>
                                            <tr className="bg-gray-100 border-b-2 border-gray-300">
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">#</th>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">Time</th>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">Timestamp (s)</th>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">Frame</th>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">Similarity</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {results.detections.map((detection, idx) => (
                                                <tr key={idx} className="border-b border-gray-200 hover:bg-gray-50">
                                                    <td className="px-4 py-3 text-gray-700">{idx + 1}</td>
                                                    <td className="px-4 py-3 text-gray-700 font-medium">
                                                        {detection.time_string}
                                                    </td>
                                                    <td className="px-4 py-3 text-gray-700">
                                                        {detection.timestamp.toFixed(2)}
                                                    </td>
                                                    <td className="px-4 py-3 text-gray-700">
                                                        {detection.frame_number}
                                                    </td>
                                                    <td className="px-4 py-3">
                                                        <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                                                            {(detection.similarity * 100).toFixed(2)}%
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>

                                {/* Download Button */}
                                <button
                                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition"
                                    onClick={handleDownloadCSV}
                                >
                                    📥 Download Detections CSV
                                </button>
                            </>
                        )}

                        {/* Segments Section */}
                        {segments.length > 0 && (
                            <>
                                <h3 className="text-xl font-semibold text-gray-800 mb-4 mt-8">🕒 Detected Time Segments</h3>

                                <div className="mb-4">
                                    <label className="block text-sm font-medium text-gray-700">
                                        Max gap between detections to merge (seconds)
                                    </label>
                                    <input
                                        type="number"
                                        min={0.1}
                                        max={5.0}
                                        step={0.1}
                                        value={gapThreshold}
                                        onChange={(e) => setGapThreshold(Number(e.target.value))}
                                        className="mt-2 p-2 border border-gray-300 rounded-lg w-full max-w-xs bg-white text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    />
                                    <p className="mt-2 text-sm text-gray-600">
                                        Current merge gap: {gapThreshold.toFixed(1)} seconds
                                    </p>
                                </div>

                                <div className="overflow-x-auto mb-4">
                                    <table className="w-full border-collapse">
                                        <thead>
                                            <tr className="bg-gray-100 border-b-2 border-gray-300">
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">#</th>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">Start Time</th>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">End Time</th>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">Duration</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {segments.map((segment, idx) => (
                                                <tr key={idx} className="border-b border-gray-200 hover:bg-gray-50">
                                                    <td className="px-4 py-3 text-gray-700">{idx + 1}</td>
                                                    <td className="px-4 py-3 text-gray-700 font-medium">
                                                        {segment.start_time}
                                                    </td>
                                                    <td className="px-4 py-3 text-gray-700">
                                                        {segment.end_time}
                                                    </td>
                                                    <td className="px-4 py-3">
                                                        <span className="px-3 py-1 bg-orange-100 text-orange-800 rounded-full text-sm font-medium">
                                                            {segment.duration}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>

                                {/* Download Segments Button */}
                                <button
                                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition"
                                    onClick={handleDownloadSegmentsCSV}
                                >
                                    📥 Download Segments CSV
                                </button>
                            </>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default Dashboard;
