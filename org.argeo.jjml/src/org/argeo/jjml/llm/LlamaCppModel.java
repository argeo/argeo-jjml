package org.argeo.jjml.llm;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.argeo.jjml.llm.params.ModelParam.n_gpu_layers;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.function.DoubleConsumer;
import java.util.function.DoublePredicate;
import java.util.function.LongSupplier;

import org.argeo.jjml.llm.params.ModelParam;
import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.util.InstructRole;

/**
 * Access to a llama.cpp model. (see <code>llama_model</code>, in llama.h)
 */
public class LlamaCppModel implements LongSupplier, AutoCloseable {

	/** The raw default model parameters as provided by libllama. */
	private final static ModelParams DEFAULT_MODEL_PARAMS_NATIVE;

	static {
		DEFAULT_MODEL_PARAMS_NATIVE = LlamaCppBackend.newModelParams();
	}

	private final long pointer;

	private final LlamaCppVocabulary vocabulary;

	private final Path localPath;

	private final ModelParams initParams;

	private boolean destroyed = false;

	// effective parameters
	private final int vocabularySize;
	private final int contextTrainingSize;
	private final int embeddingSize;
	private final int layerCount;
	private final Map<String, String> metadata;
	private final String description;
	private final long modelSize;
	private final int endOfGenerationToken;

	private String chatTemplate = null;

	LlamaCppModel(long pointer, Path localPath, ModelParams initParams) {
		this.pointer = pointer;
		this.vocabulary = new LlamaCppVocabulary(this);
		this.localPath = localPath;
		this.initParams = initParams;

		// effective parameters from native side
		vocabularySize = doGetVocabularySize();
		contextTrainingSize = doGetContextTrainingSize();
		embeddingSize = doGetEmbeddingSize();
		layerCount = doGetLayerCount();
		byte[][] keys = doGetMetadataKeys();
		byte[][] values = doGetMetadataValues();
		if (keys.length != values.length)
			throw new IllegalStateException("Metadata keys and values don't have the same size");
		LinkedHashMap<String, String> map = new LinkedHashMap<>();// preserve order
		for (int i = 0; i < keys.length; i++) {
			map.put(new String(keys[i], UTF_8), new String(values[i], UTF_8));
		}
		metadata = Collections.unmodifiableMap(map);
		if (metadata.containsKey("tokenizer.chat_template")) {
			chatTemplate = metadata.get("tokenizer.chat_template");
		}

		description = new String(doGetDescription(), UTF_8);
		modelSize = doGetModelSize();
		endOfGenerationToken = doGetEndOfGenerationToken();
	}

	/*
	 * NATIVE METHODS
	 */
	// Lifecycle
	private static native long doInit(String localPathStr, ModelParams params, DoublePredicate progressCallback);

	private native void doDestroy();

	// Accessors
	private native int doGetVocabularySize();

	private native int doGetContextTrainingSize();

	private native int doGetEmbeddingSize();

	private native int doGetLayerCount();

	private native byte[][] doGetMetadataKeys();

	private native byte[][] doGetMetadataValues();

	private native byte[] doGetDescription();

	private native long doGetModelSize();

	private native int doGetEndOfGenerationToken();

	/*
	 * USABLE METHODS
	 */
	public String formatChatMessages(LlamaCppChatMessage... messages) {
		return formatChatMessages(Arrays.asList(messages));
	}

	public String formatChatMessages(List<LlamaCppChatMessage> messages) {
		return LLamaCppNativeChatFormatter.formatChatMessages(messages, //
				(message) -> message.getRole().equals(InstructRole.USER.get()), chatTemplate);
	}

	/*
	 * LIFECYCLE
	 */
	@Override
	public void close() throws RuntimeException {
		checkDestroyed();
		doDestroy();
		destroyed = true;
	}

	private void checkDestroyed() {
		if (destroyed)
			throw new IllegalStateException("Model #" + pointer + " was already destroyed");
	}

	/*
	 * ACCESSORS
	 */
	@Override
	public long getAsLong() {
		checkDestroyed();
		return pointer;
	}

	public Path getLocalPath() {
		return localPath;
	}

	public ModelParams getInitParams() {
		return initParams;
	}

	public LlamaCppVocabulary getVocabulary() {
		return vocabulary;
	}

	public int getVocabularySize() {
		return vocabularySize;
	}

	public int getContextTrainingSize() {
		return contextTrainingSize;
	}

	public int getEmbeddingSize() {
		return embeddingSize;
	}

	public int getLayerCount() {
		return layerCount;
	}

	public Map<String, String> getMetadata() {
		return metadata;
	}

	public String getDescription() {
		return description;
	}

	public long getModelSize() {
		return modelSize;
	}

	public int getEndOfGenerationToken() {
		return endOfGenerationToken;
	}

	/*
	 * STATIC UTILITIES
	 */

	public static LlamaCppModel load(Path localPath) throws IOException {
		return load(localPath, defaultModelParams());
	}

	public static ModelParams defaultModelParams() {
		ModelParams res = DEFAULT_MODEL_PARAMS_NATIVE;

		// we disable GPU offload by default as it is too sensitive to context
		// and setting context parameters right
		res = res.with(n_gpu_layers, 0);

		for (ModelParam param : ModelParam.values()) {
			String sysProp = System.getProperty(param.asSystemProperty());
			if (sysProp != null)
				res = res.with(param, sysProp);
		}
		return res;
	}

	/**
	 * Loads a model synchronously. For more fine-grained control (following
	 * progress, cancelling, executor used) use
	 * {@link #loadAsync(Path, ModelParams, DoubleConsumer, Executor)}.
	 */
	public static LlamaCppModel load(Path localPath, ModelParams initParams) throws IOException {
		Future<LlamaCppModel> future = loadAsync(localPath, initParams, null, null);
		try {
			return future.get();
		} catch (InterruptedException | ExecutionException e) {
			throw new IOException("Cannot load model from " + localPath, e);
		}
	}

	/**
	 * Loads a model asynchronously. Loading the model can be cancelled by calling
	 * {@link Future#cancel(boolean)} with <code>true</code> on the returned
	 * {@link Future}.
	 */
	public static Future<LlamaCppModel> loadAsync(Path localPath, ModelParams initParams,
			DoubleConsumer progressCallback, Executor executor) throws IOException {
		Objects.requireNonNull(initParams);
		if (!Files.exists(localPath))
			throw new FileNotFoundException("Model path " + localPath + " does not exist.");

		FutureTask<LlamaCppModel> future = new FutureTask<>(() -> {
			checkInitParams(initParams);
			// long begin = System.currentTimeMillis();
			long pointer = doInit(localPath.toString(), initParams, (progress) -> {
				if (progressCallback != null)
					progressCallback.accept(progress);
				return !Thread.interrupted();
			});
			// logger.log(Level.INFO, "Model initialization took " +
			// (System.currentTimeMillis() - begin) + " ms");
			LlamaCppModel model = new LlamaCppModel(pointer, localPath, initParams);
			return model;
		});

		if (executor == null) {
			Thread loadingThread = new Thread(future, "Load model " + localPath);
			// don't continue loading if the JVM is shutting down
			loadingThread.setDaemon(true);
			loadingThread.start();
		} else {
			executor.execute(future);
		}
		return future;
	}

	private static void checkInitParams(ModelParams initParams) {
//		if (initParams.n_gpu_layers() != 0 && !LlamaCppBackend.supportsGpuOffload())
//			logger.log(WARNING, "GPU offload is not available, but " + ModelParam.n_gpu_layers + " is set to "
//					+ initParams.n_gpu_layers());
//		if (initParams.use_mmap() && !LlamaCppBackend.supportsMmap())
//			logger.log(WARNING,
//					"mmap is not available, but " + ModelParam.use_mmap + " is set to " + initParams.use_mmap());
//		if (initParams.use_mlock() && !LlamaCppBackend.supportsMlock())
//			logger.log(WARNING,
//					"mlock is not available, but " + ModelParam.use_mlock + " is set to " + initParams.use_mlock());
	}
}
