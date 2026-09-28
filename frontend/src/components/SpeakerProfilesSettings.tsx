'use client';

import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Button } from '@/components/ui/button';
import { Pencil, Trash2, Check, X, User } from 'lucide-react';
import { toast } from 'sonner';

interface SpeakerProfile {
  id: string;
  name: string;
  is_named: boolean;
  sample_count: number;
  sample_audio_path: string | null;
  created_at: string;
  updated_at: string;
}

/**
 * Settings section listing every speaker voiceprint diarization has learned,
 * including auto-created "Unknown Speaker #N" profiles. Lets the user rename
 * or delete a saved speaker.
 */
export function SpeakerProfilesSettings() {
  const [profiles, setProfiles] = useState<SpeakerProfile[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');

  const fetchProfiles = async () => {
    try {
      setIsLoading(true);
      const data = await invoke<SpeakerProfile[]>('diarization_list_speaker_profiles');
      setProfiles(data);
    } catch (error) {
      console.error('Failed to load speaker profiles:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchProfiles();
  }, []);

  const startEditing = (profile: SpeakerProfile) => {
    setEditingId(profile.id);
    setEditingName(profile.is_named ? profile.name : '');
  };

  const saveName = async (id: string) => {
    const trimmed = editingName.trim();
    if (!trimmed) {
      toast.error('Speaker name cannot be empty');
      return;
    }
    try {
      await invoke('diarization_rename_speaker', { speakerId: id, newName: trimmed });
      setEditingId(null);
      fetchProfiles();
      toast.success('Speaker renamed');
    } catch (error) {
      console.error('Failed to rename speaker:', error);
      toast.error('Failed to rename speaker');
    }
  };

  const deleteProfile = async (id: string, name: string) => {
    try {
      await invoke('diarization_delete_speaker_profile', { speakerId: id });
      setProfiles((prev) => prev.filter((p) => p.id !== id));
      toast.success(`Removed "${name}"`);
    } catch (error) {
      console.error('Failed to delete speaker profile:', error);
      toast.error('Failed to delete speaker');
    }
  };

  if (isLoading) {
    return <p className="text-sm text-gray-500">Loading saved speakers...</p>;
  }

  if (profiles.length === 0) {
    return (
      <p className="text-sm text-gray-500">
        No speakers identified yet. Speakers appear here automatically after a meeting with speaker
        identification enabled is processed.
      </p>
    );
  }

  return (
    <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
      {profiles.map((profile) => (
        <div
          key={profile.id}
          className="flex items-center justify-between gap-3 p-3 rounded-md border border-gray-200 bg-white"
        >
          <div className="flex items-center gap-3 min-w-0">
            <div className="h-8 w-8 rounded-full bg-gray-100 flex items-center justify-center shrink-0">
              <User className="h-4 w-4 text-gray-500" />
            </div>
            <div className="min-w-0">
              {editingId === profile.id ? (
                <input
                  autoFocus
                  value={editingName}
                  onChange={(e) => setEditingName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') saveName(profile.id);
                    if (e.key === 'Escape') setEditingId(null);
                  }}
                  className="text-sm px-2 py-1 border border-gray-300 rounded-md w-48"
                  placeholder="Speaker name"
                />
              ) : (
                <>
                  <p className={`text-sm font-medium truncate ${profile.is_named ? 'text-gray-900' : 'text-gray-500 italic'}`}>
                    {profile.name}
                  </p>
                  <p className="text-xs text-gray-400">{profile.sample_count} sample{profile.sample_count === 1 ? '' : 's'}</p>
                </>
              )}
            </div>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            {editingId === profile.id ? (
              <>
                <button
                  className="p-1.5 rounded hover:bg-gray-100 text-green-600"
                  onClick={() => saveName(profile.id)}
                  title="Save"
                >
                  <Check className="h-4 w-4" />
                </button>
                <button
                  className="p-1.5 rounded hover:bg-gray-100 text-gray-500"
                  onClick={() => setEditingId(null)}
                  title="Cancel"
                >
                  <X className="h-4 w-4" />
                </button>
              </>
            ) : (
              <>
                <button
                  className="p-1.5 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-900"
                  onClick={() => startEditing(profile)}
                  title="Rename speaker"
                >
                  <Pencil className="h-4 w-4" />
                </button>
                <button
                  className="p-1.5 rounded hover:bg-gray-100 text-gray-500 hover:text-red-600"
                  onClick={() => deleteProfile(profile.id, profile.name)}
                  title="Delete speaker"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
