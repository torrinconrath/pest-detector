import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertCircle, Shield, Camera } from "lucide-react";

const LIVE_FEED_URL = "http://10.0.0.113:8081/video_feed";
const ALERTS_URL = "http://10.0.0.113:8081/alerts";

export default function PestDetectionDashboard() {
  const [alerts, setAlerts] = useState([]);
  const [liveFeedUrl, setLiveFeedUrl] = useState(LIVE_FEED_URL);

  // // Mock alert system (replace with WebSocket/REST from Flask)
  // useEffect(() => {
  //   const interval = setInterval(() => {
  //     const threats = ["Person", "Knife", "Unknown Object"];
  //     const randomThreat = threats[Math.floor(Math.random() * threats.length)];
  //     setAlerts((prev) => [
  //       { id: Date.now(), type: randomThreat, time: new Date().toLocaleTimeString() },
  //       ...prev.slice(0, 4),
  //     ]);
  //   }, 10000);
  //   return () => clearInterval(interval);
  // }, []);

  useEffect(() => {
    let evtSource;
    let retryTimeout;

    const connect = () => {
      evtSource = new EventSource(ALERTS_URL);

      evtSource.onopen = () => {
        console.log("Connected to alert server ✅");
      };

      evtSource.onmessage = (event) => {
        try {
          const alert = JSON.parse(event.data);
          setAlerts((prev) => [
            { id: Date.now(), type: alert.type, time: alert.time || new Date().toLocaleTimeString() },
            ...prev.slice(0, 4),
          ]);
        } catch (err) {
          console.error("Failed to parse alert", err);
        }
      };

      evtSource.onerror = (err) => {
        console.error("SSE connection lost, retrying in 3s...", err);
        evtSource.close();
        retryTimeout = setTimeout(connect, 10000); // retry after 10 seconds
      };

    };

    connect();

  return () => {
    if (evtSource) evtSource.close();
    if (retryTimeout) clearTimeout(retryTimeout);
  };
}, []);


  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      {/* Header */}
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Shield className="text-green-400" /> Threat Detection Dashboard
        </h1>
        <Button className="bg-green-500 hover:bg-green-600">Settings</Button>
      </header>

      {/* Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Feed */}
        <Card className="col-span-2 bg-gray-900 border border-gray-800">
          <CardContent className="p-0">
            <div className="relative w-full aspect-video overflow-hidden rounded-2xl">
              <motion.img
                key={liveFeedUrl}
                src={liveFeedUrl}
                alt="Live Feed"
                className="w-full h-full object-cover"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1 }}
              />
              <div className="absolute top-3 left-3 bg-green-600 px-3 py-1 rounded-full text-sm font-semibold">
                LIVE
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Alerts */}
        <Card className="bg-gray-900 border border-gray-800">
          <CardContent>
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <AlertCircle className="text-red-400" /> Alerts
            </h2>
            <div className="space-y-3">
              {alerts.length === 0 ? (
                <p className="text-gray-400 text-sm">No threats detected yet.</p>
              ) : (
                alerts.map((alert) => (
                  <motion.div
                    key={alert.id}
                    className="bg-red-900/30 border border-red-500/50 rounded-xl p-3 flex justify-between items-center"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                  >
                    <span className="font-bold text-red-300">{alert.type}</span>
                    <span className="text-xs text-gray-400">{alert.time}</span>
                  </motion.div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Footer */}
      <footer className="mt-6 text-center text-gray-500 text-sm flex items-center justify-center gap-2">
        <Camera size={16} /> Raspberry Pi Video Detection Active
      </footer>
    </div>
  );
}
