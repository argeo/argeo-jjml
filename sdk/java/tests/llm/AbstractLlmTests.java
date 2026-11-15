//!/usr/bin/env -S java -ea -cp /usr/share/java/org.argeo.jjml.jar
package tests.llm;

import static java.lang.System.Logger.Level.DEBUG;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.util.SimpleProgressCallback;

import tests.AbstractJjmlTests;
import tests.HfModelCache;

/**
 * Minimal set of non-destructive in-memory tests, in order to check that a
 * given deployment and/or model are working. Java assertions must be enabled.
 */
public abstract class AbstractLlmTests extends AbstractJjmlTests {
	private int parallelism = Runtime.getRuntime().availableProcessors();

	private final LlamaCppModel model;

	public AbstractLlmTests(LlamaCppModel model) {
		Objects.requireNonNull(model);
		this.model = model;
	}

	/*
	 * ACCESSORS
	 */
	protected LlamaCppModel getModel() {
		return model;
	}

	protected int getParallelism() {
		return parallelism;
	}

	public static LlamaCppModel createModel(String hint) throws IOException {
		return createModel(LlamaCppModel.defaultModelParams(), hint);
	}

	public static LlamaCppModel createModel(ModelParams modelParams, String hint) throws IOException {
		Path modelPath = Paths.get(hint);
		if (!Files.exists(modelPath))
			modelPath = new HfModelCache().getLocalFile(hint);
		if (modelPath == null || !Files.exists(modelPath))
			throw new IllegalArgumentException(
					"Could not find GGUF model " + hint + ". Make sure that llama.cpp tools can find it.");
		return loadModel(modelParams, modelPath);
	}

	static LlamaCppModel loadModel(ModelParams modelParams, Path modelPath) throws IOException {
		SimpleProgressCallback progressCallback = System.getLogger(LoadModelTests.class.getName()).isLoggable(DEBUG)
				? new SimpleProgressCallback()
				: null;
		Future<LlamaCppModel> loaded = LlamaCppModel.loadAsync(modelPath, modelParams, progressCallback, null);
		try {
			return loaded.get();
		} catch (InterruptedException | ExecutionException e) {
			throw new IllegalStateException("Could not load model " + modelPath, e);
		}
	
	}

}
