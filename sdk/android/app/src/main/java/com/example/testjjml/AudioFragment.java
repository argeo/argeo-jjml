package com.example.testjjml;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import org.argeo.jjml.whisper.WhisperCppProcessor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

public class AudioFragment extends Fragment {

    private Button btnRecord;
    private TextView audioStatus;
    private TextView audioTranscription;
    private TextView audioModelStatus;

    private boolean isRecording = false;
    private AudioRecord audioRecord;
    private Thread recordingThread;
    private final ByteArrayOutputStream audioData = new ByteArrayOutputStream();

    private static final int SAMPLE_RATE = 16000;

    private ActivityResultLauncher<String> requestPermissionLauncher;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_audio, container, false);

        btnRecord = view.findViewById(R.id.btn_record);
        audioStatus = view.findViewById(R.id.audio_status);
        audioTranscription = view.findViewById(R.id.audio_transcription);
        audioModelStatus = view.findViewById(R.id.audio_model_status);

        // Load Whisper model when this tab is opened
        loadWhisperModel();

        updateModelStatus();

        requestPermissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(),
                isGranted -> {
                    if (isGranted) {
                        toggleRecording();
                    } else {
                        Toast.makeText(getContext(), "Microphone permission is required.", Toast.LENGTH_SHORT).show();
                    }
                }
        );

        Button btnTestAudio = view.findViewById(R.id.btn_test_audio);
        btnTestAudio.setOnClickListener(v -> testWithSampleAudio());

        btnRecord.setOnClickListener(v -> {
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                toggleRecording();
            } else {
                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO);
            }
        });

        return view;
    }

    private void loadWhisperModel() {
        if (!LlmManager.INSTANCE.isWhisperLoaded()) {
            audioModelStatus.setText("Whisper Model: Loading...");
            audioModelStatus.setTextColor(0xFF0000FF);
            LlmManager.INSTANCE.loadWhisper(requireContext(), () -> {
                updateModelStatus();
                audioModelStatus.setText("Whisper Model: Successfully loaded!");
                audioModelStatus.setTextColor(0xFF008800);
                
                // Reset to normal status after 2 seconds
                new Handler(Looper.getMainLooper()).postDelayed(() -> {
                    updateModelStatus();
                }, 2000);
            });
        }
    }

    private void updateModelStatus() {
        if (getContext() == null) return;
        boolean downloaded = LlmManager.INSTANCE.isWhisperModelDownloaded(requireContext());
        if (LlmManager.INSTANCE.isWhisperLoaded()) {
            audioModelStatus.setText("Whisper Model: Loaded & Ready");
            audioModelStatus.setTextColor(0xFF008800);
        } else if (downloaded) {
            audioModelStatus.setText("Whisper Model: Downloaded (Ready to load)");
            audioModelStatus.setTextColor(0xFF008800);
        } else {
            audioModelStatus.setText("Whisper Model: Not downloaded (will download on first recording)");
            audioModelStatus.setTextColor(0xFF888800);
        }
    }

    private void toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    @SuppressLint("MissingPermission")
    private void startRecording() {
        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize);

        audioData.reset();
        audioRecord.startRecording();
        isRecording = true;
        btnRecord.setText("Stop Recording");
        audioStatus.setText("Recording...");
        audioTranscription.setText("");

        recordingThread = new Thread(() -> {
            byte[] data = new byte[bufferSize];
            while (isRecording) {
                int read = audioRecord.read(data, 0, bufferSize);
                if (read > 0) {
                    audioData.write(data, 0, read);
                }
            }
        });
        recordingThread.start();
    }

    private void stopRecording() {
        isRecording = false;
        if (audioRecord != null) {
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;
        }
        btnRecord.setText("Start Recording");
        audioStatus.setText("Processing...");

        processAudio();
    }

    private void processAudio() {
        if (!LlmManager.INSTANCE.isWhisperLoaded()) {
            audioStatus.setText("Loading Whisper model...");
            audioModelStatus.setText("Whisper Model: Downloading/Loading...");
            audioModelStatus.setTextColor(0xFF0000FF);
            LlmManager.INSTANCE.loadWhisper(requireContext(), () -> {
                updateModelStatus();
                doProcessAudio();
            });
        } else {
            doProcessAudio();
        }
    }

    private void testWithSampleAudio() {
        audioStatus.setText("Testing with sample audio...");
        audioTranscription.setText("");
        
        // Generate a simple sine wave as test audio
        int sampleRate = 16000;
        int durationMs = 2000; // 2 seconds
        int numSamples = (sampleRate * durationMs) / 1000;
        
        // Create a simple "hello world" sine wave pattern
        float[] samples = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            double time = (double) i / sampleRate;
            // Simple sine wave at 440Hz (A4 note)
            samples[i] = (float) Math.sin(2 * Math.PI * 440 * time) * 0.3f;
        }
        
        // Convert to FloatBuffer for Whisper
        FloatBuffer floatBuffer = ByteBuffer.allocateDirect(samples.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        for (float sample : samples) {
            floatBuffer.put(sample);
        }
        floatBuffer.flip();
        
        // Process the test audio
        LlmManager.INSTANCE.executeInBg(() -> {
            try {
                WhisperCppProcessor processor = LlmManager.INSTANCE.getWhisperProcessor();
                String transcription = processor.transcribe(floatBuffer);

                new Handler(Looper.getMainLooper()).post(() -> {
                    audioStatus.setText("Test complete.");
                    audioTranscription.setText("Sample transcription: " + transcription);
                });

            } catch (Exception e) {
                e.printStackTrace();
                new Handler(Looper.getMainLooper()).post(() -> {
                    audioStatus.setText("Test error.");
                    audioTranscription.setText("Error: " + e.getMessage());
                });
            }
        });
    }

    private void doProcessAudio() {
        audioStatus.setText("Transcribing audio...");
        btnRecord.setEnabled(false);

        LlmManager.INSTANCE.executeInBg(() -> {
            try {
                byte[] pcmData = audioData.toByteArray();

                // Convert 16-bit PCM bytes to floats (-1.0 to 1.0)
                ByteBuffer byteBuffer = ByteBuffer.wrap(pcmData).order(ByteOrder.LITTLE_ENDIAN);
                ShortBuffer shortBuffer = byteBuffer.asShortBuffer();

                FloatBuffer floatBuffer = ByteBuffer.allocateDirect(shortBuffer.limit() * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
                for (int i = 0; i < shortBuffer.limit(); i++) {
                    float val = shortBuffer.get() / (float) Short.MAX_VALUE;
                    floatBuffer.put(val);
                }
                floatBuffer.flip();

                WhisperCppProcessor processor = LlmManager.INSTANCE.getWhisperProcessor();
                String transcription = processor.transcribe(floatBuffer);

                new Handler(Looper.getMainLooper()).post(() -> {
                    audioStatus.setText("Done.");
                    audioTranscription.setText(transcription);
                    btnRecord.setEnabled(true);
                });

            } catch (Exception e) {
                e.printStackTrace();
                new Handler(Looper.getMainLooper()).post(() -> {
                    audioStatus.setText("Error.");
                    audioTranscription.setText(e.getMessage());
                    btnRecord.setEnabled(true);
                });
            }
        });
    }
}
