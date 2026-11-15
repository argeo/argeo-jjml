package com.example.testjjml;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppInstructProcessor;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.params.ContextParams;
import org.argeo.jjml.llm.params.ModelParams;

import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class LlmManager {
    public static final LlmManager INSTANCE = new LlmManager();

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private LlamaCppModel model;
    private LlamaCppContext context;
    private LlamaCppSamplerChain chain;
    private LlamaCppInstructProcessor processor;

    private boolean isLoaded = false;
    private String status = "Uninitialized";

    public interface Listener {
        void onStatusUpdate(String status);
        void onLoaded();
    }

    private Listener listener;
    public void setListener(Listener l) { this.listener = l; }

    public LlamaCppInstructProcessor getProcessor() {
        return processor;
    }

    public boolean isLoaded() {
        return isLoaded;
    }

    public String getStatus() {
        return status;
    }

    public void load(Context androidContext) {
        if (isLoaded) {
            if (listener != null) listener.onLoaded();
            return;
        }

        updateStatus("Initializing...");

        executor.submit(() -> {
            try {
                Path modelsBase = Paths.get(androidContext.getExternalFilesDir(null).getAbsolutePath());
                Path modelPath = modelsBase.resolve("Qwen3.5-0.8B-Q4_K_M.gguf");

                if (!Files.exists(modelPath)) {
                    updateStatus("Downloading model...");
                    String url = "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf";
                    try (InputStream in = URI.create(url).toURL().openStream()) {
                        Files.copy(in, modelPath, StandardCopyOption.REPLACE_EXISTING);
                    }
                }

                updateStatus("Loading model into memory...");
                ModelParams modelParams = LlamaCppModel.defaultModelParams();

                ContextParams contextParams = LlamaCppContext.defaultContextParams()
                        .with(ContextParam.n_ctx, 1024)
                        .with(ContextParam.n_batch, 512)
                        .with(ContextParam.n_threads, Runtime.getRuntime().availableProcessors());

                model = LlamaCppModel.load(modelPath, modelParams);
                context = new LlamaCppContext(model, contextParams);
                chain = LlamaCppSamplers.newDefaultSampler(false);
                processor = new LlamaCppInstructProcessor(context, chain);

                isLoaded = true;
                new Handler(Looper.getMainLooper()).post(() -> {
                    if (listener != null) listener.onLoaded();
                });

            } catch (Exception e) {
                e.printStackTrace();
                updateStatus("Error: " + e.getMessage());
            }
        });
    }

    public void executeInBg(Runnable r) {
        executor.submit(r);
    }

    private void updateStatus(String s) {
        status = s;
        new Handler(Looper.getMainLooper()).post(() -> {
            if (listener != null) listener.onStatusUpdate(s);
        });
    }
}
