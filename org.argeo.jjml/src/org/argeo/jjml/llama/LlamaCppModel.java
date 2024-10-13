package org.argeo.jjml.llama;

import static java.nio.charset.StandardCharsets.UTF_8;

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
	/** Tokenize a standard UTF-8 encoded string. */
	native int[] doTokenize(byte[] str, boolean addSpecial, boolean parseSpecial);

	/** De-tokenize as a standard UTF-8 encoded string. */
	native byte[] doDeTokenize(int[] tokens, boolean removeSpecial, boolean unparseSpecial);

	native long doInit(String localPathStr, LlamaCppModelParams params, DoublePredicate progressCallback);

	@Override
	native void doDestroy();

	native String doFormatChatMessages(String[] roles, String[] contents);

	native int doGetEmbeddingSize();

	/*
	 * USABLE METHODS
	 */
	public String deTokenize(LlamaCppTokenList tokenList, boolean unparseSpecial) {
		return deTokenize(tokenList, false, unparseSpecial);
	}

	public String deTokenize(LlamaCppTokenList tokenList, boolean removeSpecial, boolean unparseSpecial) {
		byte[] str = doDeTokenize(tokenList.getTokens(), removeSpecial, unparseSpecial);
		return new String(str, UTF_8);
	}

	public LlamaCppTokenList tokenize(String str, boolean addSpecial) {
		return tokenize(str, addSpecial, false);
	}

	public LlamaCppTokenList tokenize(String str, boolean addSpecial, boolean parseSpecial) {
		int[] tokens = doTokenize(str.getBytes(UTF_8), addSpecial, parseSpecial);
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
