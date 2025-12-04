import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
//import { Button } from "@/components/ui/button";
import { AlertCircle, Camera, Server, WifiOff } from "lucide-react";
import raccoon_logo from "./assets/raccoon_logo.png"
import "./App.css";

// Two backends to toggle between
const SERVERS = {
  pi: "http://10.0.0.113:8082",
  desktop: "http://localhost:8082" // UPDATE this to your WSL IP
};

export default function PestDetectionDashboard() {
  const [mode, setMode] = useState(() => {
    return localStorage.getItem("pest_mode") || "pi"; // load mode from localStorage
  });

  // Save mode in localStorage
  useEffect(() => {
    localStorage.setItem("pest_mode", mode);
  }, [mode]);

  // Dynamic endpoint mapping
  const BASE_URL = SERVERS[mode];
  const VIDEO_FEED_URL = `${BASE_URL}/video_feed`;
  const STATUS_URL = `${BASE_URL}/status`;

  // State
  const [status, setStatus] = useState({
    active_alerts: 0,
    total_detections: 0,
    camera: "disconnected",
    model: "failed",
    last_detection: "Never",
    performance: {},
    system: {}
  });
  const [isConnected, setIsConnected] = useState(false);

  // Poll status regularly
  useEffect(() => {
    let statusInterval;

    // Reset UI when switching servers
    setStatus({
      active_alerts: 0,
      total_detections: 0,
      camera: "connecting",
      model: "loading",
      last_detection: "Never",
      performance: {},
      system: {}
    });
    setIsConnected(false);

    // Fetch status
    const fetchStatus = async () => {
      try {
        const res = await fetch(STATUS_URL);
        const data = await res.json();
        setStatus(data);
        setIsConnected(true);
      } catch (e) {
        console.error("Status fetch failed:", e);
        setIsConnected(false);
      }
    };

    fetchStatus();
    statusInterval = setInterval(fetchStatus, 2000); // Poll every other second

    return () => {
      if (statusInterval) clearInterval(statusInterval);
    };
  }, [mode]); // Re-run when mode changes

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">

      {/* Header */}
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-5">
          <img src={raccoon_logo} alt="logo" className="h-8 w-8"/> Pest Detection Dashboard
        </h1>

        <div className="flex items-center gap-4">

          {/* Mode Toggle */}
          <div className="flex items-center gap-2 bg-gray-800 px-3 py-1 rounded-full">
            <Server size={18} className="text-blue-400" />
            <select
              className="bg-transparent text-white outline-none"
              value={mode}
              onChange={(e) => setMode(e.target.value)}
            >
              <option value="pi" className="bg-white text-black">Raspberry Pi Mode</option>
              <option value="desktop" className="bg-white text-black">WSL / Desktop Mode</option>
            </select>
          </div>

          {/* Connection pill */}
          <div className={`flex items-center gap-2 text-sm px-3 py-1 rounded-full ${
            isConnected ? 'bg-green-500' : 'bg-red-500'
          }`}>
            <div className="w-2 h-2 rounded-full bg-white" />
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </header>

      {/* Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Video Feed */}
        <Card className="col-span-2 bg-gray-900 border border-gray-800">
          <CardContent className="p-0">
            <div className="relative w-full aspect-video overflow-hidden rounded-2xl bg-black flex items-center justify-center">
              {isConnected ? (
                <>
                  <motion.img
                    key={mode}
                    src={VIDEO_FEED_URL}
                    alt="Live Pest Detection Feed"
                    className="w-full h-full object-contain"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.8 }}
                  />
                  <div className="absolute top-3 left-3 bg-green-600 px-3 py-1 rounded-full text-sm font-semibold">
                    LIVE
                  </div>
                </>
              ) : (
                <div className="text-gray-500 flex flex-col items-center gap-2">
                  <WifiOff size={48} />
                  <span className="text-lg font-semibold">Camera Disconnected</span>
                  <span className="text-sm">Please check server connection.</span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Alerts & Status */}
        <Card className="bg-gray-900 border border-gray-800">
          <CardContent>
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 mt-3 text-white">
              <AlertCircle className="text-red-400" />
              Detection Status
            </h2>
            {isConnected ? (
              <div className="space-y-4">
                {/* Current Alert */}
                {status.active_alerts > 0 ? (
                  <motion.div
                    className="bg-red-900/30 border border-red-500/50 rounded-xl p-4"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-bold text-red-300 text-lg">PEST DETECTED!</span>
                      <span className="text-sm text-gray-400">{status.last_detection}</span>
                    </div>
                    <div className="text-sm text-gray-300">
                      Active threats being monitored
                    </div>
                  </motion.div>
                ) : (
                  <div className="bg-green-900/30 border border-green-500/50 rounded-xl p-4">
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-bold text-green-300 text-lg">NO PEST!</span>
                      <span className="text-sm text-gray-400">
                        {status.last_detection === "Never" ? "No detections" : `Last: ${status.last_detection}`}
                      </span>
                    </div>
                    <div className="text-sm text-gray-300">
                      No active threats detected
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4 text-center text-gray-500 flex flex-col items-center gap-2">
                <Server size={32} />
                <span className="text-lg font-semibold">Server Disconnected</span>
                <span className="text-sm">Cannot retrieve detection status.</span>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Status Bar */}
      <footer className="mt-6 flex items-center justify-between text-sm text-gray-400">
        <div className="flex items-center gap-4">
          <Camera size={16} />
          <span>{mode === "pi" ? "Pi Inference Active" : "Desktop Inference Active"}</span>
          <span>Active Alerts: {status.active_alerts}</span>
          <span>Total Detections: {status.total_detections}</span>

          <span>Processing Time: {(status.performance.preprocess_ms != null && status.performance.process_ms != null) 
            ? `${(status.performance.preprocess_ms + status.performance.process_ms)?.toFixed(2)} ms` : 'N/A'}</span>

          <span>Inference Time: {status.performance.inference_ms != null 
            ? `${status.performance.inference_ms?.toFixed(2)} ms` : 'N/A'}</span>

          <span>Total Time: {status.performance.total_ms != null
            ? `${status.performance.total_ms?.toFixed(2)} ms` : 'N/A'}</span>
        </div>
        <div className="text-xs">
          Last update: {status.timestamp || "--:--:--"}
        </div>
      </footer>
    </div>
  );
}