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

	native int[] doTokenize(String str, boolean addSpecial, boolean parseSpecial);

	native String doDeTokenize(int[] tokens, boolean special);

	native long doInit(String localPathStr, LlamaCppModelParams params, DoublePredicate progressCallback);

	@Override
	native void doDestroy();

	native String doFormatChatMessages(String[] roles, String[] contents);

	native int doGetEmbeddingSize();

	/*
	 * USABLE METHODS
	 */
	public String deTokenize(LlamaCppTokenList tokenList, boolean special) {
		return doDeTokenize(tokenList.getTokens(), special);
	}

	public LlamaCppTokenList tokenize(String str, boolean addSpecial) {
		return tokenize(str, addSpecial, false);
	}

	public LlamaCppTokenList tokenize(String str, boolean addSpecial, boolean parseSpecial) {
		int[] tokens = doTokenize(str, addSpecial, parseSpecial);
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
