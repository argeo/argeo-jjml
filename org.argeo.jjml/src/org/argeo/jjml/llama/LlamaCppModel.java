package org.argeo.jjml.llama;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
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
import org.argeo.jjml.llama.params.ModelParams;

/**
 * Access to a llama.cpp model
 * 
 * @see llama.h - llama_model
 */
public class LlamaCppModel implements LongSupplier, AutoCloseable {
	private final static Logger logger = System.getLogger(LlamaCppModel.class.getName());

	private final static ModelParams DEFAULT_PARAMS;

	static {
//		DEFAULT_PARAMS = ModelParams.defaultModelParams();
		LlamaCppNative.ensureLibrariesLoaded();
		DEFAULT_PARAMS = LlamaCppNative.newModelParams();

	}

	private final long pointer;

	private final LlamaCppVocabulary vocabulary;

	private final Path localPath;

	private final ModelParams initParams;

	private final int embeddingSize;

	private boolean destroyed = false;

	LlamaCppModel(long pointer, Path localPath, ModelParams initParams) {
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
	private static native long doInit(String localPathStr, ModelParams params, DoublePredicate progressCallback);

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

	public ModelParams getInitParams() {
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

	public static ModelParams defaultModelParams() {
		return DEFAULT_PARAMS;
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

}
