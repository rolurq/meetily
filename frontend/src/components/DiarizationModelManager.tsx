'use client';

import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { cn } from '@/lib/utils';
import { Download, RefreshCw, Trash2, CheckCircle2 } from 'lucide-react';
import { toast } from 'sonner';

interface DiarizationModelInfo {
  name: string;
  path: string;
  size_mb: number;
  purpose: string;
  description: string;
  status:
    | 'Available'
    | 'Missing'
    | { Downloading: { progress: number } }
    | { Error: string };
}

function statusType(status: DiarizationModelInfo['status']): 'available' | 'missing' | 'downloading' | 'error' {
  if (status === 'Available') return 'available';
  if (status === 'Missing') return 'missing';
  if (typeof status === 'object' && 'Downloading' in status) return 'downloading';
  return 'error';
}

/**
 * Lets the user download one or more speaker-identification (diarization) models
 * and pick which one is currently active. Mirrors BuiltInModelManager's UX for
 * downloading built-in models, but for voiceprint extraction instead of LLM weights.
 */
export function DiarizationModelManager() {
  const [models, setModels] = useState<DiarizationModelInfo[]>([]);
  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [downloadingModels, setDownloadingModels] = useState<Set<string>>(new Set());
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});
  const [loadingModel, setLoadingModel] = useState<string | null>(null);

  const fetchModels = async () => {
    try {
      setIsLoading(true);
      const [data, current] = await Promise.all([
        invoke<DiarizationModelInfo[]>('diarization_get_available_models'),
        invoke<string | null>('diarization_get_current_model'),
      ]);
      setModels(data);
      setCurrentModel(current);
    } catch (error) {
      console.error('Failed to fetch diarization models:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    listen('diarization-model-download-progress', (event: any) => {
      const { modelName, progress, status } = event.payload;
      setDownloadProgress((prev) => ({ ...prev, [modelName]: progress }));
      if (status === 'downloading') {
        setDownloadingModels((prev) => new Set(prev).add(modelName));
      }
    }).then((fn) => (unlisten = fn));
    return () => unlisten?.();
  }, []);

  useEffect(() => {
    let unlistenComplete: (() => void) | undefined;
    let unlistenError: (() => void) | undefined;

    listen('diarization-model-download-complete', (event: any) => {
      const { modelName } = event.payload;
      setDownloadingModels((prev) => {
        const next = new Set(prev);
        next.delete(modelName);
        return next;
      });
      toast.success(`Speaker identification model "${modelName}" downloaded`);
      fetchModels();
    }).then((fn) => (unlistenComplete = fn));

    listen('diarization-model-download-error', (event: any) => {
      const { modelName, error } = event.payload;
      setDownloadingModels((prev) => {
        const next = new Set(prev);
        next.delete(modelName);
        return next;
      });
      toast.error(`Failed to download "${modelName}"`, { description: error });
    }).then((fn) => (unlistenError = fn));

    return () => {
      unlistenComplete?.();
      unlistenError?.();
    };
  }, []);

  const downloadModel = async (modelName: string) => {
    setDownloadingModels((prev) => new Set(prev).add(modelName));
    try {
      await invoke('diarization_download_model', { modelName });
    } catch (error) {
      console.error('Failed to download diarization model:', error);
      setDownloadingModels((prev) => {
        const next = new Set(prev);
        next.delete(modelName);
        return next;
      });
    }
  };

  const cancelDownload = async (modelName: string) => {
    await invoke('diarization_cancel_download', { modelName });
    setDownloadingModels((prev) => {
      const next = new Set(prev);
      next.delete(modelName);
      return next;
    });
  };

  const deleteModel = async (modelName: string) => {
    try {
      await invoke('diarization_delete_model', { modelName });
      toast.success(`Deleted "${modelName}"`);
      fetchModels();
    } catch (error) {
      toast.error(`Failed to delete "${modelName}"`);
    }
  };

  const useModel = async (modelName: string) => {
    setLoadingModel(modelName);
    try {
      await invoke('diarization_load_model', { modelName });
      setCurrentModel(modelName);
      toast.success(`Using "${modelName}" for speaker identification`);
    } catch (error) {
      console.error('Failed to load diarization model:', error);
      toast.error('Failed to activate model');
    } finally {
      setLoadingModel(null);
    }
  };

  if (isLoading && models.length === 0) {
    return (
      <div className="text-center py-6 text-muted-foreground">
        <RefreshCw className="mx-auto h-6 w-6 animate-spin mb-2" />
        Loading speaker identification models...
      </div>
    );
  }

  return (
    <div>
      <div className="mb-3">
        <h4 className="text-sm font-bold text-gray-900">Speaker Identification Models</h4>
        <p className="text-xs text-gray-500 mt-1">
          Download a voice-recognition model to automatically tell speakers apart in your recordings.
          Every processed meeting will be diarized using whichever model is active below.
        </p>
      </div>

      <div className="grid gap-3">
        {models.map((model) => {
          const type = statusType(model.status);
          const progress = downloadProgress[model.name];
          const isDownloading = downloadingModels.has(model.name) || type === 'downloading';
          const isActive = currentModel === model.name;

          return (
            <div
              key={model.name}
              className={cn(
                'p-4 rounded-lg border transition-colors bg-card',
                isActive ? 'ring-2 ring-gray-800 border-gray-800' : 'border-gray-200'
              )}
            >
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-gray-900">{model.purpose}</span>
                    <span className="text-xs text-gray-500">({model.size_mb} MB)</span>
                    {type === 'available' && (
                      <span className="flex items-center gap-1 text-xs font-medium text-green-600">
                        <span className="h-2 w-2 rounded-full bg-green-600" />
                        Ready
                      </span>
                    )}
                    {isActive && (
                      <span className="rounded bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
                        Active
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{model.description}</p>
                </div>
                <div className="flex shrink-0 flex-wrap items-center gap-2">
                  {type === 'missing' && !isDownloading && (
                    <Button variant="outline" size="sm" onClick={() => downloadModel(model.name)}>
                      <Download className="mr-2 h-4 w-4" />
                      Download
                    </Button>
                  )}
                  {isDownloading && (
                    <Button variant="outline" size="sm" onClick={() => cancelDownload(model.name)}>
                      Cancel
                    </Button>
                  )}
                  {type === 'error' && !isDownloading && (
                    <Button variant="outline" size="sm" onClick={() => downloadModel(model.name)}>
                      <RefreshCw className="mr-2 h-4 w-4" />
                      Retry
                    </Button>
                  )}
                  {type === 'available' && !isActive && (
                    <>
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={loadingModel === model.name}
                        onClick={() => useModel(model.name)}
                      >
                        {loadingModel === model.name ? (
                          <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                          <CheckCircle2 className="mr-2 h-4 w-4" />
                        )}
                        Use this model
                      </Button>
                      <button
                        className="p-2 rounded hover:bg-gray-100 text-gray-500 hover:text-red-600"
                        onClick={() => deleteModel(model.name)}
                        title="Delete model"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </>
                  )}
                </div>
              </div>

              {isDownloading && progress !== undefined && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-900">Downloading...</span>
                    <span className="text-sm font-semibold text-gray-900">{Math.round(progress)}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-gray-800 to-gray-900 rounded-full transition-all"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}

              {type === 'error' && typeof model.status === 'object' && 'Error' in model.status && (
                <Alert className="mt-3 border-red-200">
                  <AlertDescription className="text-xs text-red-600">{model.status.Error}</AlertDescription>
                </Alert>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
