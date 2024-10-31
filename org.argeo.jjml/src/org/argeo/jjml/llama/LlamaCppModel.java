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

import org.argeo.jjml.llama.LlamaCppChatMessage.StandardRole;

/**
 * Access to a llama.cpp model
 * 
 * @see llama.h - llama_model
 */
public class LlamaCppModel implements LongSupplier, AutoCloseable {
	private final static Logger logger = System.getLogger(LlamaCppModel.class.getName());

	public final static Params DEFAULT_PARAMS;

	static {
		assert Params.assertParamNames();
		DEFAULT_PARAMS = Params.defaultModelParams();
	}

	private final long pointer;

	private final Path localPath;

	private final Params initParams;

	private final int embeddingSize;

	LlamaCppModel(long pointer, Path localPath, Params initParams) {
		this.pointer = pointer;
		this.localPath = localPath;
		this.initParams = initParams;
		this.embeddingSize = doGetEmbeddingSize();
	}

	/*
	 * NATIVE METHODS
	 */
	// Lifcycle

	private static native long doInit(String localPathStr, Params params, DoublePredicate progressCallback);

	native void doDestroy();

	// Tokenization
	/**
	 * Tokenize a string encoded as a standard UTF-8 byte array. To use when it
	 * makes more sense to convert on the Java side.
	 * 
	 * @see #doDeTokenizeAsUtf8Array(int[], boolean, boolean)
	 */
	native int[] doTokenizeUtf8Array(byte[] str, boolean addSpecial, boolean parseSpecial);

	/** De-tokenize as a string encoded in standard UTF-8. */
	native byte[] doDeTokenizeAsUtf8Array(int[] tokens, boolean removeSpecial, boolean unparseSpecial);

	/**
	 * Tokenize a Java {@link String}. Its UTF-16 representation will be used
	 * without copy on the native side, where it will be converted to UTF-8.
	 */
	native int[] doTokenizeString(String str, boolean addSpecial, boolean parseSpecial);

	/** De-tokenize as a Java {@link String}. */
	native String doDeTokenizeAsString(int[] tokens, boolean removeSpecial, boolean unparseSpecial);

	// Chat
	native String doFormatChatMessages(String[] roles, String[] contents, boolean addAssistantTokens);

	native int doGetEmbeddingSize();

	/*
	 * USABLE METHODS
	 */
	public String deTokenize(LlamaCppTokenList tokenList, boolean unparseSpecial) {
		return deTokenize(tokenList, false, unparseSpecial);
	}

	public String deTokenize(LlamaCppTokenList tokenList, boolean removeSpecial, boolean unparseSpecial) {
//		byte[] str = doDeTokenizeAsUtf8Array(tokenList.getTokens(), removeSpecial, unparseSpecial);
//		return new String(str, UTF_8);
		return doDeTokenizeAsString(tokenList.getTokens(), removeSpecial, unparseSpecial);
	}

	public LlamaCppTokenList tokenize(String str, boolean addSpecial) {
		return tokenize(str, addSpecial, true);
	}

	public LlamaCppTokenList tokenize(String str, boolean addSpecial, boolean parseSpecial) {
//		int[] tokens = doTokenizeUtf8Array(str.getBytes(UTF_8), addSpecial, parseSpecial);
		int[] tokens = doTokenizeString(str, addSpecial, parseSpecial);
		return new LlamaCppTokenList(this, tokens);
	}

	public String formatChatMessages(List<LlamaCppChatMessage> messages) {
		String[] roles = new String[messages.size()];
		String[] contents = new String[messages.size()];

		boolean currIsUserRole = false;
		for (int i = 0; i < messages.size(); i++) {
			LlamaCppChatMessage message = messages.get(i);
			roles[i] = message.getRole();
			currIsUserRole = message.getRole().equals(StandardRole.USER.get());
			contents[i] = message.getContent();
		}

		String res = doFormatChatMessages(roles, contents, currIsUserRole);
		return res;
	}

	/*
	 * LIFECYCLE
	 */
	/** Init model with defaults. */
//	public void init() {
//		init(Params.defaultModelParams(), null);
//	}
//
//	public void init(Map<String, Object> properties) {
//		Map<ParamName, String> map = properties.entrySet().stream().filter((entry) -> {
//			try {
//				ParamName.valueOf(entry.getKey());
//			} catch (IllegalArgumentException e) {
//				return false;
//			}
//			return true;
//		}).collect(Collectors.toMap( //
//				e -> ParamName.valueOf(e.getKey()), //
//				e -> e.getValue().toString() //
//		));
//		init(DEFAULT_PARAMS.with(map), null);
//	}

//	public void init(Params modelParams, DoublePredicate progressCallback) {
//		checkNotInitialized();
//		Objects.requireNonNull(localPath, "Local path to the model must be set");
//		if (!Files.exists(localPath))
//			throw new IllegalArgumentException("Model file does not exist: " + localPath);
//		long pointer = doInit(localPath.toString(), modelParams, progressCallback);
//		setPointer(pointer);
//		this.initParams = modelParams;
//		embeddingSize = doGetEmbeddingSize();
//	}

	@Override
	public void close() throws RuntimeException {
		doDestroy();
	}

	/*
	 * ACESSORS
	 */
	@Override
	public long getAsLong() {
		return pointer;
	}

	public Path getLocalPath() {
		return localPath;
	}

//	public void setLocalPath(Path localPath) {
//		this.localPath = localPath;
//	}

	public Params getInitParams() {
		return initParams;
	}

	public int getEmbeddingSize() {
		return embeddingSize;
	}

	/*
	 * STATIC UTILITIES
	 */
	public static LlamaCppModel load(Path localPath) throws IOException {
		return load(localPath, DEFAULT_PARAMS);
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
