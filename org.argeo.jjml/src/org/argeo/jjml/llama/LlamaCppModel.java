package org.argeo.jjml.llama;

import static java.lang.Boolean.parseBoolean;
import static java.lang.Integer.parseInt;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.lang.reflect.RecordComponent;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
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
import java.util.function.Predicate;

import org.argeo.jjml.llama.LlamaCppChatMessage.StandardRole;

/**
 * Access to a llama.cpp model
 * 
 * @see llama.h - llama_model
 */
public class LlamaCppModel implements LongSupplier, AutoCloseable {
	private final static Logger logger = System.getLogger(LlamaCppModel.class.getName());

	private final static Params DEFAULT_PARAMS;

	static {
		assert Params.assertParamNames();
		DEFAULT_PARAMS = Params.defaultModelParams();
	}

	private final long pointer;

	private final LlamaCppVocabulary vocabulary;

	private final Path localPath;

	private final Params initParams;

	private final int embeddingSize;

	private boolean destroyed = false;

	LlamaCppModel(long pointer, Path localPath, Params initParams) {
		this.pointer = pointer;
		this.vocabulary = new LlamaCppVocabulary(this);
		this.localPath = localPath;
		this.initParams = initParams;
		this.embeddingSize = doGetEmbeddingSize();
	}

	/*
	 * NATIVE METHODS
	 */
	// Chat
	private native String doFormatChatMessages(long pointer, String[] roles, String[] contents,
			boolean addAssistantTokens);

	// Lifecycle
	private static native long doInit(String localPathStr, Params params, DoublePredicate progressCallback);

	private native void doDestroy();

	// Accessors
	private native int doGetEmbeddingSize();

	/*
	 * USABLE METHODS
	 */
	public String formatChatMessages(LlamaCppChatMessage... messages) {
		return formatChatMessages(Arrays.asList(messages));
	}

	public String formatChatMessages(List<LlamaCppChatMessage> messages) {
		return formatChatMessages(messages, //
				(message) -> message.getRole().equals(StandardRole.USER.get()));
	}

	public String formatChatMessages(List<LlamaCppChatMessage> messages,
			Predicate<LlamaCppChatMessage> addAssistantTokens) {
		String[] roles = new String[messages.size()];
		String[] contents = new String[messages.size()];

		boolean currIsUserRole = false;
		for (int i = 0; i < messages.size(); i++) {
			LlamaCppChatMessage message = messages.get(i);
			roles[i] = message.getRole();
			currIsUserRole = addAssistantTokens.test(message);
			contents[i] = message.getContent();
		}

		String res = doFormatChatMessages(pointer, roles, contents, currIsUserRole);
		return res;
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
	 * ACESSORS
	 */
	@Override
	public long getAsLong() {
		checkDestroyed();
		return pointer;
	}

	public Path getLocalPath() {
		return localPath;
	}

	public Params getInitParams() {
		return initParams;
	}

	public int getEmbeddingSize() {
		return embeddingSize;
	}

	public LlamaCppVocabulary getVocabulary() {
		return vocabulary;
	}

	/*
	 * STATIC UTILITIES
	 */

	public static LlamaCppModel load(Path localPath) throws IOException {
		return load(localPath, DEFAULT_PARAMS);
	}

	public static Params defaultModelParams() {
		return DEFAULT_PARAMS;
	}

	/**
	 * Loads a model synchronously. For more fine-grained control (following
	 * progress, cancelling, executor used) use
	 * {@link #loadAsync(Path, Params, DoubleConsumer, Executor)}.
	 */
	public static LlamaCppModel load(Path localPath, Params initParams) throws IOException {
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
	public static Future<LlamaCppModel> loadAsync(Path localPath, Params initParams, DoubleConsumer progressCallback,
			Executor executor) throws IOException {
		Objects.requireNonNull(initParams);
		if (!Files.exists(localPath))
			throw new FileNotFoundException("Model path " + localPath + " does not exist.");

		LlamaCppBackend.ensureInitialized();

		FutureTask<LlamaCppModel> future = new FutureTask<>(() -> {
			long begin = System.currentTimeMillis();
			long pointer = doInit(localPath.toString(), initParams, (progress) -> {
				if (progressCallback != null)
					progressCallback.accept(progress);
				return !Thread.interrupted();
			});
			logger.log(Level.INFO, "Model initialization took " + (System.currentTimeMillis() - begin) + " ms");
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

	/*
	 * CLASSES
	 */
	/** Names of the supported model parameters. */
	public static enum ParamName {
		n_gpu_layers, //
		vocab_only, //
		use_mlock, //
		;
	}

	/**
	 * Initialization parameters of a model. New instance should be created by using
	 * the {@link #with(Map)} methods on {@link LlamaCppModel#DEFAULT_PARAMS}, with
	 * the default values populated by the shared library.
	 * 
	 * @see llama.h - llama_model_params
	 */
	public static record Params(int n_gpu_layers, boolean vocab_only, boolean use_mlock) {
		public Params with(ParamName key, Object value) {
			Objects.requireNonNull(key);
			Objects.requireNonNull(value);
			return with(Collections.singletonMap(key, value.toString()));
		}

		public Params with(Map<ParamName, String> p) {
			return new Params( //
					parseInt(p.getOrDefault(ParamName.n_gpu_layers, Integer.toString(this.n_gpu_layers))), //
					parseBoolean(p.getOrDefault(ParamName.vocab_only, Boolean.toString(this.vocab_only))), //
					parseBoolean(p.getOrDefault(ParamName.use_mlock, Boolean.toString(this.use_mlock))) //
			);
		}

		/**
		 * The default model parameters.
		 * 
		 * @see llama.h - llama_model_default_params()
		 */
		private static Params defaultModelParams() {
			LlamaCppNative.ensureLibrariesLoaded();
			return LlamaCppNative.newModelParams();
		}

		/** Ensure that components and enum are perfectly in line. */
		private static boolean assertParamNames() {
			RecordComponent[] components = Params.class.getRecordComponents();
			ParamName[] names = ParamName.values();
			assert components.length == names.length;
			for (int i = 0; i < components.length; i++) {
				assert components[i].getName().equals(names[i].name());
			}
			return true;
		}
	}

}
