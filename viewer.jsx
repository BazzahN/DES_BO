import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

// Assumes your PNG files are served in /public/images/<type>_Plot_<iter>.png
// Example: /images/GP_Plot_1.png
// Adjust paths as needed.

const plotTypes = [
  { key: "GP", label: "GP Plot", prefix: "GP_Plot_" },
  { key: "Uncer", label: "Uncertainty", prefix: "Uncer_Plot_" },
  { key: "AF", label: "Acquisition", prefix: "AF_Plot_" },
  { key: "Improv", label: "Improvement", prefix: "improv_plot_" }
];

export default function PlotSeriesViewer() {
  const [iteration, setIteration] = useState(1);
  const [maxIter, setMaxIter] = useState(100); // configurable
  const [playing, setPlaying] = useState(false);
  const [activeTab, setActiveTab] = useState("GP");

  useEffect(() => {
    let interval = null;
    if (playing) {
      interval = setInterval(() => {
        setIteration((prev) => (prev < maxIter ? prev + 1 : 1));
      }, 500);
    }
    return () => clearInterval(interval);
  }, [playing, maxIter]);

  const currentPlot = plotTypes.find((p) => p.key === activeTab);
  const imagePath = currentPlot
    ? `${currentPlot.prefix}${iteration}.png`
    : "";

  return (
    <div className="p-6 space-y-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Simulation Plot Series Viewer</h1>

      <div className="flex space-x-4 items-center">
        <label className="font-medium">Iteration:</label>
        <input
          type="number"
          className="border rounded p-2 w-24"
          min={1}
          max={maxIter}
          value={iteration}
          onChange={(e) => setIteration(Number(e.target.value))}
        />

        <label className="font-medium ml-4">Max Iter:</label>
        <input
          type="number"
          className="border rounded p-2 w-24"
          min={1}
          value={maxIter}
          onChange={(e) => setMaxIter(Number(e.target.value))}
        />

        <Button onClick={() => setPlaying(!playing)}>
          {playing ? "Pause" : "Play"}
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList>
          {plotTypes.map((t) => (
            <TabsTrigger key={t.key} value={t.key}>
              {t.label}
            </TabsTrigger>
          ))}
        </TabsList>

        {plotTypes.map((t) => (
          <TabsContent key={t.key} value={t.key}>
            <div className="flex justify-center mt-4">
              <img
                src={`${t.prefix}${iteration}.png`}
                alt={`${t.label} iteration ${iteration}`}
                className="rounded-xl shadow-xl max-h-[600px]"
              />
            </div>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
