package org.argeo.jjml.llama;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.function.DoublePredicate;

/**
 * Access to a llama.cpp model
 * 
 * @see llama.h - llama_model
 */
public class LlamaCppModel extends NativeReference {
	private Path localPath;

	private LlamaCppModelParams initParams;

	private int embeddingSize;

	public LlamaCppModel() {
		LlamaCppBackend.ensureInitialized();
	}

	/*
	 * NATIVE METHODS
	 */
	// Lifcycle

	native long doInit(String localPathStr, LlamaCppModelParams params, DoublePredicate progressCallback);

	@Override
	native void doDestroy(long pointer);

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
	native String doFormatChatMessages(String[] roles, String[] contents);

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
		return tokenize(str, addSpecial, false);
	}

	public LlamaCppTokenList tokenize(String str, boolean addSpecial, boolean parseSpecial) {
		// int[] tokens = doTokenizeUtf8Array(str.getBytes(UTF_8), addSpecial,
		// parseSpecial);
		int[] tokens = doTokenizeString(str, addSpecial, parseSpecial);
		return new LlamaCppTokenList(this, tokens);
	}

	public String formatChatMessages(List<LlamaCppChatMessage> messages) {
		String[] roles = new String[messages.size()];
		String[] contents = new String[messages.size()];

		for (int i = 0; i < messages.size(); i++) {
			LlamaCppChatMessage message = messages.get(i);
			roles[i] = message.role();
			contents[i] = message.content();
		}

		String res = doFormatChatMessages(roles, contents);
		return res;
	}

	/*
	 * LIFECYCLE
	 */
	/** Init model with defaults. */
	public void init() {
		init(LlamaCppModelParams.defaultModelParams(), null);
	}

	public void init(LlamaCppModelParams modelParams, DoublePredicate progressCallback) {
		checkNotInitialized();
		Objects.requireNonNull(localPath, "Local path to the model must be set");
		if (!Files.exists(localPath))
			throw new IllegalArgumentException("Model file does not exist: " + localPath);
		long pointer = doInit(localPath.toString(), modelParams, progressCallback);
		setPointer(pointer);
		this.initParams = modelParams;
		embeddingSize = doGetEmbeddingSize();

//		String str = doDeTokenizeAsString(null, false, false);
//		System.out.println(str);
	}

	/*
	 * ACESSORS
	 */

	public Path getLocalPath() {
		return localPath;
	}

//	String getLocalPathAsString() {
//		Objects.requireNonNull(localPath);
//		return localPath.toString();
//	}

	public void setLocalPath(Path localPath) {
		this.localPath = localPath;
	}

	public LlamaCppModelParams getInitParams() {
		return initParams;
	}

	public int getEmbeddingSize() {
		return embeddingSize;
	}
}
