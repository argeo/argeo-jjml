package com.example.testjjml;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import org.argeo.jjml.llm.LlamaCppNativeChatFormatter;
import org.argeo.jjml.llm.LlamaCppChatMessage;
import org.argeo.jjml.llm.util.InstructRole;
import org.argeo.jjml.mtmd.MtmdBackend;
import org.argeo.jjml.mtmd.MtmdBitmap;
import org.argeo.jjml.mtmd.MtmdProcessor;

import java.io.IOException;

public class ImageFragment extends Fragment {

    private ImageView imagePreview;
    private Button btnTakePicture;
    private TextView imageDescription;
    private TextView imageModelStatus;

    private ActivityResultLauncher<Intent> takePictureLauncher;
    private Bitmap currentBitmap;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_image, container, false);

        imagePreview = view.findViewById(R.id.image_preview);
        btnTakePicture = view.findViewById(R.id.btn_take_picture);
        imageDescription = view.findViewById(R.id.image_description);
        imageModelStatus = view.findViewById(R.id.image_model_status);

        // Load MTMD model when this tab is opened
        loadMtmdModel();

        updateModelStatus();

        takePictureLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                        Bundle extras = result.getData().getExtras();
                        if (extras != null && extras.get("data") != null) {
                            currentBitmap = (Bitmap) extras.get("data");
                            imagePreview.setImageBitmap(currentBitmap);
                            processImage();
                        }
                    }
                }
        );

        Button btnTestImage = view.findViewById(R.id.btn_test_image);
        btnTestImage.setOnClickListener(v -> testWithSampleImage());

        btnTakePicture.setOnClickListener(v -> {
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (takePictureIntent.resolveActivity(requireActivity().getPackageManager()) != null) {
                takePictureLauncher.launch(takePictureIntent);
            }
        });

        return view;
    }

    private void loadMtmdModel() {
        if (!LlmManager.INSTANCE.isMtmdLoaded()) {
            imageModelStatus.setText("MMProj Model: Loading...");
            imageModelStatus.setTextColor(0xFF0000FF);
            LlmManager.INSTANCE.loadMtmd(requireContext(), () -> {
                updateModelStatus();
                imageModelStatus.setText("MMProj Model: Successfully loaded!");
                imageModelStatus.setTextColor(0xFF008800);
                
                // Reset to normal status after 2 seconds
                new Handler(Looper.getMainLooper()).postDelayed(() -> {
                    updateModelStatus();
                }, 2000);
            });
        }
    }

    private void updateModelStatus() {
        if (getContext() == null) return;
        boolean downloaded = LlmManager.INSTANCE.isMmprojDownloaded(requireContext());
        if (LlmManager.INSTANCE.isMtmdLoaded()) {
            imageModelStatus.setText("MMProj Model: Loaded & Ready");
            imageModelStatus.setTextColor(0xFF008800);
        } else if (downloaded) {
            imageModelStatus.setText("MMProj Model: Downloaded (Ready to load)");
            imageModelStatus.setTextColor(0xFF008800);
        } else {
            imageModelStatus.setText("MMProj Model: Not downloaded (will download on first photo)");
            imageModelStatus.setTextColor(0xFF888800);
        }
    }

    private void processImage() {
        if (currentBitmap == null) return;

        if (!LlmManager.INSTANCE.isMtmdLoaded()) {
            imageDescription.setText("Loading MTMD model...");
            imageModelStatus.setText("MMProj Model: Downloading/Loading...");
            imageModelStatus.setTextColor(0xFF0000FF);
            LlmManager.INSTANCE.loadMtmd(requireContext(), () -> {
                updateModelStatus();
                doProcessImage();
            });
        } else {
            doProcessImage();
        }
    }

    private void testWithSampleImage() {
        imageDescription.setText("Testing with sample image...");
        
        // Create a simple test pattern as a bitmap
        Bitmap testBitmap = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(testBitmap);
        Paint paint = new Paint();
        
        // Draw a simple pattern for testing
        paint.setColor(Color.RED);
        canvas.drawCircle(100, 100, 80, paint);
        paint.setColor(Color.BLUE);
        canvas.drawText("Test Image", 70, 120, paint);
        
        currentBitmap = testBitmap;
        imagePreview.setImageBitmap(currentBitmap);
        processImage();
    }

    private void doProcessImage() {
        imageDescription.setText("Describing image...");
        btnTakePicture.setEnabled(false);

        LlmManager.INSTANCE.executeInBg(() -> {
            try {
                MtmdProcessor processor = LlmManager.INSTANCE.getMtmdProcessor();

                LlamaCppChatMessage systemPrompt = null;
                LlamaCppChatMessage userPrompt = new LlamaCppChatMessage(InstructRole.USER,
                        MtmdBackend.getDefaultMarker() + "Describe this image in detail.");

                String formatted = new LlamaCppNativeChatFormatter(
                        LlmManager.INSTANCE.getModel().getMetadataChatTemplate())
                        .formatChatMessages(systemPrompt, userPrompt);

                AndroidBitmap bitmap = AndroidBitmap.fromBitmap(currentBitmap);
                MtmdBitmap[] bitmaps = new MtmdBitmap[] { bitmap };

                String response = processor.transcribe(formatted, bitmaps);

                new Handler(Looper.getMainLooper()).post(() -> {
                    imageDescription.setText(response);
                    btnTakePicture.setEnabled(true);
                });
            } catch (Exception e) {
                e.printStackTrace();
                new Handler(Looper.getMainLooper()).post(() -> {
                    imageDescription.setText("Error: " + e.getMessage());
                    btnTakePicture.setEnabled(true);
                });
            }
        });
    }
}
