package com.example.testjjml;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

import com.example.testjjml.databinding.ActivityMainBinding;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppInstructProcessor;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNative;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.params.ContextParams;
import org.argeo.jjml.llm.params.ModelParam;
import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.util.InstructRole;
import org.argeo.jjml.mtmd.MtmdNative;
import org.argeo.jjml.whisper.WhisperNative;

import java.io.InputStream;
import java.io.StringWriter;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

public class MainActivity extends AppCompatActivity {

    static {
        LlamaCppNative.ensureLibrariesLoaded();
        System.out.println("MTMD: " + MtmdNative.isAvailable());
        System.out.println("Whisper: " + WhisperNative.isAvailable());
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        TextView tv = binding.sampleText;

        new Thread(()-> {
            long begin = System.currentTimeMillis();
            try {
                Path modelsBase = Paths.get(getExternalFilesDir(null).getAbsolutePath());
                boolean isWritable = Files.isWritable(modelsBase);
                Path modelPath = modelsBase.resolve("granite-4.0-350m-Q4_0.gguf");
                if (!Files.exists(modelPath)) {
                    String url = "https://huggingface.co/ibm-granite/granite-4.0-350m-GGUF/resolve/main/granite-4.0-350m-Q4_0.gguf";
                    System.out.println("Downloading " + url);
                    try (InputStream in = URI.create(url).toURL().openStream()) {
                        Files.copy(in, modelPath, StandardCopyOption.REPLACE_EXISTING);
                    }
                }
                
                ModelParams modelParams = LlamaCppModel.defaultModelParams() //
                        .with(ModelParam.vocab_only, false) //
                        ;
                ContextParams contextParams = LlamaCppContext.defaultContextParams() //
                        .with(ContextParam.n_ctx, 1024) //
                        .with(ContextParam.n_batch, 512) //
                        .with(ContextParam.n_threads, Runtime.getRuntime().availableProcessors()) //
                        ;
                System.out.println("LOADING...");
                System.out.flush();
                try (LlamaCppModel model = LlamaCppModel.load(modelPath,modelParams); //
                     LlamaCppContext context = new LlamaCppContext(model,contextParams); //
                     LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
                ) {
                    System.out.println("WRITING...");
                    LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);
                    processor.write(InstructRole.SYSTEM, "You are a helpful assistant.");
                    processor.write(InstructRole.USER, "In one word, the capital of France is ");
                    System.out.println("READING...");
                    StringWriter out = new StringWriter();
                    //Writer out = new OutputStreamWriter(System.out, StandardCharsets.UTF_8);
                    processor.readMessage(out);
                    System.out.println(out.toString());
                    System.out.println("DONE in "+(System.currentTimeMillis() - begin) / 1000 + " s");
                }
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }).start();
        tv.setText("Supports GPU offload: " + LlamaCppBackend.supportsGpuOffload());
    }

    /**
     * A native method that is implemented by the 'testnative' native library,
     * which is packaged with this application.
     */
    //public native String stringFromJNI();
}