'use client';

import { useEffect, useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Button } from '@/components/ui/button';
import { Play, Pause, User, X } from 'lucide-react';
import { toast } from 'sonner';

interface DiarizedSpeaker {
  speaker_id: string;
  name: string;
  is_named: boolean;
  confidence: number;
  sample_audio_path: string | null;
}

interface SpeakerProfile {
  id: string;
  name: string;
  is_named: boolean;
}

const NEW_SPEAKER_OPTION = '__new__';

function SpeakerRow({
  meetingId,
  speaker,
  profiles,
  onResolved,
}: {
  meetingId: string;
  speaker: DiarizedSpeaker;
  profiles: SpeakerProfile[];
  onResolved: (speakerId: string) => void;
}) {
  const [selection, setSelection] = useState<string>(speaker.speaker_id);
  const [newName, setNewName] = useState(speaker.is_named ? speaker.name : '');
  const [saving, setSaving] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    let objectUrl: string | null = null;
    if (speaker.sample_audio_path) {
      invoke<number[]>('diarization_read_sample_audio', { path: speaker.sample_audio_path })
        .then((bytes) => {
          const blob = new Blob([new Uint8Array(bytes)], { type: 'audio/wav' });
          objectUrl = URL.createObjectURL(blob);
          setAudioUrl(objectUrl);
        })
        .catch((err) => console.error('Failed to load speaker sample:', err));
    }
    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [speaker.sample_audio_path]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
  };

  const save = async () => {
    setSaving(true);
    try {
      if (selection === NEW_SPEAKER_OPTION) {
        if (!newName.trim()) {
          toast.error('Enter a name for this speaker');
          setSaving(false);
          return;
        }
        await invoke('diarization_reassign_meeting_speaker', {
          meetingId,
          fromSpeakerId: speaker.speaker_id,
          newSpeakerName: newName.trim(),
        });
      } else if (selection !== speaker.speaker_id) {
        await invoke('diarization_reassign_meeting_speaker', {
          meetingId,
          fromSpeakerId: speaker.speaker_id,
          targetSpeakerId: selection,
        });
      } else if (newName.trim() && newName.trim() !== speaker.name) {
        await invoke('diarization_rename_speaker', { speakerId: speaker.speaker_id, newName: newName.trim() });
      }
      onResolved(speaker.speaker_id);
    } catch (error) {
      console.error('Failed to save speaker identification:', error);
      toast.error('Failed to save speaker');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex items-center gap-3 p-3 rounded-lg border border-gray-200 bg-white">
      <div className="h-9 w-9 rounded-full bg-gray-100 flex items-center justify-center shrink-0">
        <User className="h-4 w-4 text-gray-500" />
      </div>

      {audioUrl && (
        <>
          <audio
            ref={audioRef}
            src={audioUrl}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onEnded={() => setIsPlaying(false)}
          />
          <button
            onClick={togglePlay}
            className="h-8 w-8 rounded-full bg-gray-800 text-white flex items-center justify-center shrink-0 hover:bg-gray-700"
            title="Play sample"
          >
            {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5 ml-0.5" />}
          </button>
        </>
      )}

      <div className="flex-1 min-w-0 flex flex-wrap items-center gap-2">
        <select
          className="text-sm px-2 py-1.5 border border-gray-300 rounded-md bg-white"
          value={selection}
          onChange={(e) => setSelection(e.target.value)}
        >
          <option value={speaker.speaker_id}>
            {speaker.is_named ? speaker.name : `${speaker.name} (unidentified)`}
          </option>
          {profiles
            .filter((p) => p.id !== speaker.speaker_id)
            .map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          <option value={NEW_SPEAKER_OPTION}>+ New speaker...</option>
        </select>

        {(selection === NEW_SPEAKER_OPTION || (selection === speaker.speaker_id && !speaker.is_named)) && (
          <input
            className="text-sm px-2 py-1.5 border border-gray-300 rounded-md flex-1 min-w-[140px]"
            placeholder="Speaker name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
        )}

        {selection === speaker.speaker_id && speaker.is_named && speaker.confidence < 1 && (
          <span className="text-xs text-gray-500">{Math.round(speaker.confidence * 100)}% confident</span>
        )}
      </div>

      <Button size="sm" onClick={save} disabled={saving}>
        {saving ? 'Saving...' : 'Confirm'}
      </Button>
    </div>
  );
}

/**
 * Listens for the backend's `diarization-complete` event and shows a panel for
 * naming/correcting every speaker detected in the meeting that just finished
 * processing. Mounted once at the app root, like the download-progress toast provider.
 */
export function SpeakerIdentificationProvider() {
  const [meetingId, setMeetingId] = useState<string | null>(null);
  const [speakers, setSpeakers] = useState<DiarizedSpeaker[]>([]);
  const [profiles, setProfiles] = useState<SpeakerProfile[]>([]);
  const [resolved, setResolved] = useState<Set<string>>(new Set());

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    listen('diarization-complete', (event: any) => {
      const { meetingId: mid, speakers: detected } = event.payload as {
        meetingId: string;
        speakers: DiarizedSpeaker[];
      };
      if (!detected || detected.length === 0) return;
      setMeetingId(mid);
      setSpeakers(detected);
      setResolved(new Set());
      invoke<SpeakerProfile[]>('diarization_list_speaker_profiles')
        .then((list) => setProfiles(list.map((p) => ({ id: p.id, name: p.name, is_named: p.is_named }))))
        .catch(() => setProfiles([]));
    }).then((fn) => (unlisten = fn));
    return () => unlisten?.();
  }, []);

  if (!meetingId || speakers.length === 0) return null;

  const allResolved = speakers.every((s) => resolved.has(s.speaker_id));

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full max-h-[85vh] overflow-hidden flex flex-col">
        <div className="flex justify-between items-center p-5 border-b">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Who spoke in this meeting?</h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Play each sample and confirm who it is. Unnamed speakers stay as "Unknown Speaker" until you name them.
            </p>
          </div>
          <button onClick={() => setMeetingId(null)} className="text-gray-400 hover:text-gray-700">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {speakers.map((speaker) => (
            <SpeakerRow
              key={speaker.speaker_id}
              meetingId={meetingId}
              speaker={speaker}
              profiles={profiles}
              onResolved={(id) => setResolved((prev) => new Set(prev).add(id))}
            />
          ))}
        </div>

        <div className="border-t p-4 flex justify-end">
          <Button variant={allResolved ? 'default' : 'outline'} onClick={() => setMeetingId(null)}>
            {allResolved ? 'Done' : 'Close'}
          </Button>
        </div>
      </div>
    </div>
  );
}
