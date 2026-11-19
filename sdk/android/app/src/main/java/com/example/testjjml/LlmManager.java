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
import org.argeo.jjml.mtmd.MtmdContext;
import org.argeo.jjml.mtmd.MtmdProcessor;
import org.argeo.jjml.whisper.WhisperCppContext;
import org.argeo.jjml.whisper.WhisperCppProcessor;

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

    private MtmdContext mtmdContext;
    private MtmdProcessor mtmdProcessor;

    private WhisperCppContext whisperContext;
    private WhisperCppProcessor whisperProcessor;

    private boolean isLoaded = false;
    private boolean isMtmdLoaded = false;
    private boolean isWhisperLoaded = false;

    private String status = "Uninitialized";

    public interface Listener {
        void onStatusUpdate(String status);
        void onLoaded();
    }

    private Listener listener;
    public void setListener(Listener l) { this.listener = l; }

    public LlamaCppModel getModel() {
        return model;
    }

    public LlamaCppInstructProcessor getProcessor() {
        return processor;
    }

    public MtmdProcessor getMtmdProcessor() {
        return mtmdProcessor;
    }

    public WhisperCppProcessor getWhisperProcessor() {
        return whisperProcessor;
    }

    public boolean isLoaded() {
        return isLoaded;
    }

    public boolean isMtmdLoaded() {
        return isMtmdLoaded;
    }

    public boolean isWhisperLoaded() {
        return isWhisperLoaded;
    }

    public boolean isMmprojDownloaded(Context context) {
        try {
            Path modelsBase = Paths.get(context.getExternalFilesDir(null).getAbsolutePath());
            return Files.exists(modelsBase.resolve("mmproj-BF16.gguf"));
        } catch (Exception e) {
            return false;
        }
    }

    public boolean isWhisperModelDownloaded(Context context) {
        try {
            Path modelsBase = Paths.get(context.getExternalFilesDir(null).getAbsolutePath());
            return Files.exists(modelsBase.resolve("ggml-base.bin"));
        } catch (Exception e) {
            return false;
        }
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
                    updateStatus("Downloading main model...");
                    String url = "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf";
                    try (InputStream in = URI.create(url).toURL().openStream()) {
                        Files.copy(in, modelPath, StandardCopyOption.REPLACE_EXISTING);
                    }
                }

                // Download mmproj model in parallel
                downloadModelIfNeeded(androidContext, "mmproj-BF16.gguf", "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-BF16.gguf");
                
                // Download whisper model in parallel
                downloadModelIfNeeded(androidContext, "ggml-base.bin", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin");

                updateStatus("Loading main model into memory...");
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

    public void loadMtmd(Context androidContext, Runnable onLoaded) {
        if (isMtmdLoaded) {
            onLoaded.run();
            return;
        }

        executor.submit(() -> {
            try {
                Path modelsBase = Paths.get(androidContext.getExternalFilesDir(null).getAbsolutePath());
                Path mmprojPath = modelsBase.resolve("mmproj-BF16.gguf");

                if (!Files.exists(mmprojPath)) {
                    String url = "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-BF16.gguf";
                    try (InputStream in = URI.create(url).toURL().openStream()) {
                        Files.copy(in, mmprojPath, StandardCopyOption.REPLACE_EXISTING);
                    }
                }

                // wait for main model if not loaded yet
                while (!isLoaded) {
                    Thread.sleep(100);
                }

                mtmdContext = new MtmdContext(model, mmprojPath, Runtime.getRuntime().availableProcessors());
                mtmdProcessor = new MtmdProcessor(context, chain, mtmdContext);

                isMtmdLoaded = true;
                new Handler(Looper.getMainLooper()).post(onLoaded);

            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    public void loadWhisper(Context androidContext, Runnable onLoaded) {
        if (isWhisperLoaded) {
            onLoaded.run();
            return;
        }

        executor.submit(() -> {
            try {
                Path modelsBase = Paths.get(androidContext.getExternalFilesDir(null).getAbsolutePath());
                Path whisperPath = modelsBase.resolve("ggml-base.bin");

                if (!Files.exists(whisperPath)) {
                    String url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";
                    try (InputStream in = URI.create(url).toURL().openStream()) {
                        Files.copy(in, whisperPath, StandardCopyOption.REPLACE_EXISTING);
                    }
                }

                whisperContext = new WhisperCppContext(whisperPath);
                whisperProcessor = new WhisperCppProcessor(whisperContext);

                isWhisperLoaded = true;
                new Handler(Looper.getMainLooper()).post(onLoaded);

            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    private void downloadModelIfNeeded(Context androidContext, String filename, String url) {
        executor.submit(() -> {
            try {
                Path modelsBase = Paths.get(androidContext.getExternalFilesDir(null).getAbsolutePath());
                Path modelPath = modelsBase.resolve(filename);

                if (!Files.exists(modelPath)) {
                    try (InputStream in = URI.create(url).toURL().openStream()) {
                        Files.copy(in, modelPath, StandardCopyOption.REPLACE_EXISTING);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
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
